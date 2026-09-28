"""Load raw SBA FOIA files, harmonise the 7(a)/504 schemas, build the default label, split by vintage."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# Different FOIA extracts spell some headers differently -> map to one canonical name.
COLUMN_ALIASES = {
    "terminmonths": "TermInMonths",
    "termInmonths": "TermInMonths",
    "terminmonths ": "TermInMonths",
    "paidinfulldate": "PaidInFullDate",
    "fixedorvariableinterestind": "FixedOrVariableInterestInd",
    "grosschargeoffamount": "GrossChargeOffAmount",
    "chargeoffdate": "ChargeOffDate",
    "thirdpartydollars": "ThirdPartyDollars",
    "borrstate": "BorrState",
    "processingmethod": "ProcessingMethod",
    "businessage": "BusinessAge",
    "businesstype": "BusinessType",
    "nonprofit": "NonProfit",
}
CANONICAL = ["Program", "ApprovalDate", "FirstDisbursementDate", "GrossApproval", "SBAGuaranteedApproval",
             "TermInMonths", "InitialInterestRate", "FixedOrVariableInterestInd", "ProcessingMethod",
             "NaicsCode", "BorrState", "BusinessType", "BusinessAge", "CollateralInd", "FranchiseCode",
             "NonProfit", "JobsSupported", "ThirdPartyDollars", "LoanStatus", "ChargeOffDate",
             "GrossChargeOffAmount", "PaidInFullDate", "SoldSecMrktInd", "AsOfDate"]
DATE_COLS = ["ApprovalDate", "FirstDisbursementDate", "ChargeOffDate", "PaidInFullDate"]
NUMERIC_COLS = ["GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate",
                "JobsSupported", "ThirdPartyDollars", "GrossChargeOffAmount"]


def find_raw_files(path: str | Path | None = None) -> list[Path]:
    """Real FOIA CSVs win over the synthetic stand-in if both exist."""
    if path is not None:
        p = Path(path)
        return sorted(p.glob("*.csv")) if p.is_dir() else [p]
    real = sorted(config.DATA_REAL.glob("*.csv")) if config.DATA_REAL.exists() else []
    if real:
        return real
    return sorted(config.DATA_SYNTH.glob("*.csv")) if config.DATA_SYNTH.exists() else []


def _harmonise_columns(df: pd.DataFrame, program_hint: str | None) -> pd.DataFrame:
    lower = {c: c.strip() for c in df.columns}
    df = df.rename(columns=lower)
    ren = {}
    for c in df.columns:
        key = c.lower()
        if key in COLUMN_ALIASES:
            ren[c] = COLUMN_ALIASES[key]
        else:  # match canonical names case-insensitively
            for canon in CANONICAL:
                if key == canon.lower():
                    ren[c] = canon
    df = df.rename(columns=ren)
    if "Program" not in df.columns:
        df["Program"] = program_hint or "7A"
    for c in CANONICAL:  # optional columns absent in some extracts -> NaN, documented in data dictionary
        if c not in df.columns:
            df[c] = np.nan
    df["Program"] = df["Program"].astype(str).str.upper().str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    df["Program"] = np.where(df["Program"].str.contains("504"), "504", "7A")
    return df


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read + concatenate every raw CSV, returning only the canonical columns (typed)."""
    files = find_raw_files(path)
    if not files:
        raise FileNotFoundError(
            "No raw data found. Run `python -m src.data_download` (real SBA data) or "
            "`python -m src.synthetic` (synthetic stand-in). See README.")
    parts = []
    for f in files:
        hint = "504" if "504" in f.name.lower() else "7A"
        part = pd.read_csv(f, low_memory=False, encoding_errors="replace")
        parts.append(_harmonise_columns(part, hint)[CANONICAL])
        log.info("loaded %s (%d rows)", f.name, len(part))
    df = pd.concat(parts, ignore_index=True)
    for c in DATE_COLS + ["AsOfDate"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["NaicsCode"] = df["NaicsCode"].astype("string").str.replace(r"\.0$", "", regex=True)
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def build_target(df: pd.DataFrame, obs_months: int = config.OBS_MONTHS,
                 as_of: str | pd.Timestamp = config.AS_OF_DATE) -> tuple[pd.DataFrame, dict]:
    """Construct the binary default flag and drop loans whose label is not yet knowable.

    RULE (documented in docs/DATA_DICTIONARY.md):
      keep    : LoanStatus in {PIF, CHGOFF, EXEMPT} AND FirstDisbursementDate is not null
                AND FirstDisbursementDate + obs_months <= as_of   (full observation window elapsed)
      default : LoanStatus == 'CHGOFF' AND ChargeOffDate <= FirstDisbursementDate + obs_months
      Everything else kept -> 0.  Charge-offs *after* the window are 0 for this target (documented).
    Dropped: CANCLD (never funded), COMMIT (not yet funded), no disbursement date (cannot anchor
    the window), and right-censored recent loans.
    """
    as_of = pd.Timestamp(as_of)
    n0 = len(df)
    stats = {"raw_rows": n0}
    df = df[df["LoanStatus"].isin(["PIF", "CHGOFF", "EXEMPT"])].copy()
    stats["after_status_filter"] = len(df)
    df = df[df["FirstDisbursementDate"].notna() & df["ApprovalDate"].notna()].copy()
    stats["after_disbursement_filter"] = len(df)
    window_end = df["FirstDisbursementDate"] + pd.DateOffset(months=obs_months)
    df = df[window_end <= as_of].copy()
    window_end = window_end.loc[df.index]
    stats["after_censoring_filter"] = len(df)
    is_co = (df["LoanStatus"] == "CHGOFF") & df["ChargeOffDate"].notna()
    df["default"] = (is_co & (df["ChargeOffDate"] <= window_end)).astype(int)
    df["approval_year"] = df["ApprovalDate"].dt.year
    stats["late_chargeoffs_labelled_0"] = int((is_co & (df["default"] == 0)).sum())
    stats["default_rate"] = float(df["default"].mean())
    return df.reset_index(drop=True), stats


def split_by_vintage(df: pd.DataFrame, year_col: str = "approval_year"):
    """Out-of-time split on approval year -> (train, valid, test)."""
    def sel(rng):
        return df[(df[year_col] >= rng[0]) & (df[year_col] <= rng[1])].copy()
    return sel(config.TRAIN_YEARS), sel(config.VALID_YEARS), sel(config.TEST_YEARS)


def load_modeling_table(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """raw -> labelled loans (all columns still present; features.py selects the leakage-safe ones)."""
    return build_target(load_raw(path))
