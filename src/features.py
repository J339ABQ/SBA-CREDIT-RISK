"""Leakage-controlled feature engineering + sklearn preprocessing.

Design rule: a feature may only use information that exists on the *approval date*. The registry
`LEAKAGE_EXCLUDED` lists every raw column deliberately kept out, with the reason; `assert_no_leakage`
is enforced in code and in the test-suite so nobody can add one silently.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# ---------------------------------------------------------------------------------------------
# FEATURES EXCLUDED DUE TO LEAKAGE RISK, AND WHY
# ---------------------------------------------------------------------------------------------
LEAKAGE_EXCLUDED: dict[str, str] = {
    "LoanStatus": "Outcome itself (PIF / CHGOFF / ...). Encodes the label directly.",
    "ChargeOffDate": "Only exists once the loan has defaulted; the target is derived from it.",
    "GrossChargeOffAmount": "Loss amount realised after default; non-null only for defaulted loans.",
    "PaidInFullDate": "Only exists after the loan has repaid - a 'survivor' marker, known post-origination.",
    "SoldSecMrktInd": "Secondary-market sale happens after origination, and investors buy performing loans.",
    "AsOfDate": "Extract date, not a borrower attribute; constant per file and correlates with loan age.",
    "FirstDisbursementDate": "Used ONLY to anchor the 36-month outcome window, never as a feature: "
                             "disbursement happens after approval (approval->disbursement lag is post-decision).",
    "ApprovalDate/approval_year": "Used ONLY for the out-of-time split and to de-trend the rate; the raw "
                                  "calendar year is not a feature (a model must not memorise vintages).",
    "BorrName / address / bank identifiers": "Free-text / identifiers: memorisation risk, no generalisable signal, "
                                             "PII-adjacent. (Lender identity is a legitimate future feature - see README.)",
    "Loan age, remaining balance, delinquency, servicing flags": "Not in the origination record; would only exist post-approval.",
}

NUMERIC_FEATURES = ["log_gross_approval", "guarantee_pct", "term_months", "interest_rate", "rate_spread",
                    "jobs_supported", "jobs_per_100k", "third_party_ratio"]
CATEGORICAL_FEATURES = ["program", "processing_method", "naics_sector", "borrower_state", "business_type",
                        "business_age", "collateral_ind", "rate_type", "is_franchise", "non_profit"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "default"

# columns that must never appear among model inputs
_FORBIDDEN = {"LoanStatus", "ChargeOffDate", "GrossChargeOffAmount", "PaidInFullDate", "SoldSecMrktInd",
              "AsOfDate", "FirstDisbursementDate", "default", "approval_year", "ApprovalDate"}


def assert_no_leakage(columns) -> None:
    bad = sorted(set(columns) & _FORBIDDEN)
    if bad:
        raise ValueError(f"Leakage guard: forbidden columns in feature set: {bad}")


def _prime_for(year: pd.Series) -> pd.Series:
    lo, hi = min(config.PRIME_RATE), max(config.PRIME_RATE)
    return year.clip(lo, hi).map(config.PRIME_RATE).astype(float)


def _clean(v, upper: bool):
    if v is None or (not isinstance(v, str) and pd.isna(v)) or str(v).strip() == "":
        return np.nan
    v = str(v).strip()
    v = v.upper() if upper else v
    # LightGBM forbids JSON-special chars (commas, quotes...) in feature names -> normalise values
    return re.sub(r"[^0-9A-Za-z]+", "_", v).strip("_")


def _cat(s: pd.Series, upper: bool = True) -> pd.Series:
    """Text column -> object dtype with NaN for blanks; safe even when the whole column is missing."""
    return pd.Series([_clean(v, upper) for v in s], index=s.index, dtype="object")


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical raw columns -> model features (origination-time information only)."""
    X = pd.DataFrame(index=raw.index)
    gross = pd.to_numeric(raw["GrossApproval"], errors="coerce")
    gross = gross.where(gross > 0)
    guar = pd.to_numeric(raw["SBAGuaranteedApproval"], errors="coerce")
    rate = pd.to_numeric(raw["InitialInterestRate"], errors="coerce")
    jobs = pd.to_numeric(raw["JobsSupported"], errors="coerce")
    third = pd.to_numeric(raw["ThirdPartyDollars"], errors="coerce")
    year = pd.to_datetime(raw["ApprovalDate"], errors="coerce").dt.year

    X["log_gross_approval"] = np.log(gross)
    X["guarantee_pct"] = (guar / gross).clip(0, 1)
    X["term_months"] = pd.to_numeric(raw["TermInMonths"], errors="coerce")
    X["interest_rate"] = rate
    X["rate_spread"] = rate - _prime_for(year)                 # de-trended by the prime rate in force
    X["jobs_supported"] = jobs
    X["jobs_per_100k"] = jobs / (gross / 1e5)
    X["third_party_ratio"] = third / gross                     # 504 only -> NaN for 7(a)

    X["program"] = _cat(raw["Program"])
    X["processing_method"] = _cat(raw["ProcessingMethod"])
    X["naics_sector"] = _cat(raw["NaicsCode"]).map(lambda v: v[:2] if isinstance(v, str) and len(v) >= 2 else np.nan)
    X["borrower_state"] = _cat(raw["BorrState"])
    X["business_type"] = _cat(raw["BusinessType"])
    X["business_age"] = _cat(raw["BusinessAge"], upper=False)
    X["collateral_ind"] = _cat(raw["CollateralInd"])
    X["rate_type"] = _cat(raw["FixedOrVariableInterestInd"])
    X["is_franchise"] = np.where(_cat(raw["FranchiseCode"]).notna(), "Y", "N").astype(object)
    X["non_profit"] = np.where(_cat(raw["NonProfit"]).isin(["Y", "YES", "TRUE"]), "Y", "N").astype(object)
    X = X[FEATURES]
    assert_no_leakage(X.columns)
    return X


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Fit-on-train-only preprocessing (it lives inside the Pipeline, so CV/OOT never see leakage)."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), NUMERIC_FEATURES), ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
