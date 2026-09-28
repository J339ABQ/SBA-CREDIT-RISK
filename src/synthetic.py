"""Synthetic stand-in for the SBA 7(a)/504 FOIA extracts.

WHY THIS EXISTS
    The sandbox this project was built in blocks outbound access to sba.gov / data.sba.gov
    (HTTP 403 from the egress policy), so the real FOIA files could not be downloaded.
    To keep the whole pipeline runnable and demonstrable, this module generates loan-level
    data that mimics the FOIA *schema* (same column names/value vocabularies) and broad
    stylised facts (start-ups riskier, restaurants/construction riskier, long-term real-estate
    style loans safer, post-crisis and COVID vintages worse ...).

    It is NOT real SBA data. The default process is my own construction, so every metric in
    this repo computed on it is a demonstration of the *method*, not evidence about real SBA
    borrowers. Re-run on the real files (see README) to get real numbers.

Usage:  python -m src.synthetic --n-per-year 9000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
          "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
          "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
          "VT", "VA", "WA", "WV", "WI", "WY"]
# rough population weights so big states dominate (CA, TX, FL, NY ...)
_STATE_W = np.array([1.5, .2, 2, 1, 12, 1.8, 1.2, .3, .3, 6.5, 3, .4, .6, 4, 2.2, 1, 1, 1.4, 1.4, .4, 1.8,
                     2.1, 3, 1.7, .9, 1.9, .3, .6, 1, .4, 2.8, .6, 6, 3.2, .2, 3.5, 1.2, 1.3, 3.8, .3, 1.5,
                     .3, 2, 8.5, 1, .2, 2.6, 2.3, .5, 1.7, .2])

# 2-digit NAICS sector: (sampling weight, logit effect)
SECTORS = {
    "72": (14, 0.45), "23": (9, 0.25), "44": (11, 0.15), "45": (4, 0.15), "62": (9, -0.35),
    "54": (8, -0.20), "81": (10, 0.05), "31": (3, -0.10), "32": (2, -0.10), "33": (4, -0.10),
    "48": (4, 0.20), "49": (1, 0.20), "53": (3, -0.30), "11": (2, 0.00), "56": (4, 0.15),
    "61": (2, -0.10), "71": (3, 0.30), "42": (3, -0.05), "52": (1, 0.00), "51": (1, 0.05),
    "22": (0.3, -0.2), "21": (0.3, 0.1), "55": (0.2, 0.0), "92": (0.1, 0.0),
}
# vintage (approval year) shifts in the log-odds of default: post-crisis hangover, benign
# mid-cycle, COVID shock. This is the "economic cycle drift" the OOT split is designed to expose.
VINTAGE_SHIFT = {2010: .35, 2011: .20, 2012: .05, 2013: -.05, 2014: -.15, 2015: -.20, 2016: -.15,
                 2017: -.10, 2018: -.05, 2019: 0.0, 2020: .40, 2021: .10, 2022: .15, 2023: .20, 2024: .20}
VOLUME_MULT = {2010: .7, 2011: .75, 2012: .85, 2013: .9, 2014: .95, 2015: 1.0, 2016: 1.05, 2017: 1.1,
               2018: 1.15, 2019: 1.15, 2020: .95, 2021: 1.35, 2022: 1.1, 2023: .9, 2024: .9}

BUSINESS_AGE = ["Existing or more than 2 years old", "New Business or 2 years or less",
                "Startup, Loan Funds will Open Business", "Change of Ownership", "Unanswered"]
BUS_AGE_EFFECT = {BUSINESS_AGE[0]: -0.25, BUSINESS_AGE[1]: 0.35, BUSINESS_AGE[2]: 0.65,
                  BUSINESS_AGE[3]: -0.05, BUSINESS_AGE[4]: 0.10}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_per_year: int = 9000, seed: int = config.RANDOM_STATE, target_rate: float = 0.075) -> pd.DataFrame:
    """Return a FOIA-schema DataFrame of approvals FY2010-FY2024 with outcomes observed as of AS_OF_DATE."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(config.AS_OF_DATE)
    state_eff = dict(zip(STATES, rng.normal(0, 0.15, len(STATES))))
    sec_codes = list(SECTORS)
    sec_p = np.array([SECTORS[s][0] for s in sec_codes], float)
    sec_p /= sec_p.sum()

    frames = []
    for year in range(2010, 2025):
        n = int(n_per_year * VOLUME_MULT[year])
        df = pd.DataFrame(index=range(n))
        # -- program / channel
        df["Program"] = np.where(rng.random(n) < 0.14, "504", "7A")
        is504 = (df["Program"] == "504").to_numpy()
        df["ProcessingMethod"] = np.where(
            is504, "504", rng.choice(["PLP", "Regular", "Express", "Other"], n, p=[.42, .30, .20, .08]))
        # -- dates
        days = rng.integers(0, 365, n)
        df["ApprovalDate"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(days, unit="D")
        lag = rng.gamma(2.0, 30, n).astype(int).clip(0, 300)  # approval -> first disbursement (days)
        df["FirstDisbursementDate"] = df["ApprovalDate"] + pd.to_timedelta(lag, unit="D")
        # -- borrower
        df["BorrState"] = rng.choice(STATES, n, p=_STATE_W / _STATE_W.sum())
        df["NaicsCode"] = [c + f"{rng.integers(0, 9999):04d}"[: 4] for c in rng.choice(sec_codes, n, p=sec_p)]
        sector = df["NaicsCode"].str[:2]
        df["BusinessType"] = rng.choice(["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"], n, p=[.62, .28, .10])
        startup_boost = 1 + 0.25 * (year >= 2018)  # more start-ups in later vintages (mix drift)
        p_age = np.array([.50, .20 * startup_boost, .10 * startup_boost, .17, .03])
        df["BusinessAge"] = rng.choice(BUSINESS_AGE, n, p=p_age / p_age.sum())
        df["FranchiseCode"] = np.where(rng.random(n) < 0.11, rng.integers(10000, 99999, n).astype(str), None)
        df["NonProfit"] = np.where(rng.random(n) < 0.015, "Y", None)
        # -- loan structure
        loan_scale = 1.03 ** (year - 2010)  # size inflation over time
        size_log = rng.normal(11.9, 1.05, n) + np.log(loan_scale) + 0.9 * is504 + 0.5 * (sector.eq("53")).to_numpy()
        gross = np.round(np.exp(size_log), -2).clip(5_000, 5_000_000)
        df["GrossApproval"] = gross
        gpct = np.where(gross <= 150_000, .85, .75)
        gpct = np.where(rng.random(n) < .15, .5, gpct)
        gpct = np.where(is504, 0.5, gpct)  # 504: SBA debenture is 40% (CDC share) - approximated
        df["SBAGuaranteedApproval"] = np.round(gross * gpct, -2)
        term_choices = rng.choice([60, 84, 120, 180, 240, 300], n, p=[.10, .25, .28, .07, .22, .08])
        term = np.where(is504, rng.choice([120, 240, 300], n, p=[.12, .68, .20]), term_choices)
        term = np.where((gross > 400_000) & ~is504 & (rng.random(n) < .5), 300, term)
        df["TermInMonths"] = term
        df["FixedOrVariableInterestInd"] = np.where(is504, "F", rng.choice(["V", "F"], n, p=[.72, .28]))
        df["CollateralInd"] = rng.choice(["Y", "N"], n, p=[.72, .28])
        df["JobsSupported"] = np.round(np.exp(rng.normal(1.6, .9, n)) * (gross / 250_000) ** .3).clip(0, 500)
        df["ThirdPartyDollars"] = np.where(is504, np.round(gross * rng.uniform(1.0, 1.6, n), -2), np.nan)

        # -- latent default risk (log-odds)
        startup = df["BusinessAge"].map(BUS_AGE_EFFECT).to_numpy()
        z = (
            startup
            + sector.map(lambda s: SECTORS[s][1]).to_numpy()
            + df["BorrState"].map(state_eff).to_numpy()
            - 0.16 * (np.log(gross) - 11.9)                      # bigger loans a bit safer
            - 0.0028 * np.clip(term, 0, 300) * (~is504)          # long term (RE-backed) safer
            - 0.30 * (df["CollateralInd"] == "Y").to_numpy()
            - 0.35 * is504
            + 0.20 * (df["FranchiseCode"].notna()).to_numpy()
            + 0.10 * (df["BusinessType"] == "INDIVIDUAL").to_numpy()
            - 0.25 * (df["ProcessingMethod"] == "PLP").to_numpy()
            + 0.25 * (df["ProcessingMethod"] == "Express").to_numpy()
            - 0.20 * (df["NonProfit"] == "Y").to_numpy()
            - 0.05 * np.log1p(df["JobsSupported"].to_numpy())
            # interactions the linear model can't capture: start-up restaurants & short-term small loans
            + 0.45 * (df["BusinessAge"] == BUSINESS_AGE[2]).to_numpy() * sector.eq("72").to_numpy()
            + 0.35 * ((term <= 84) & (gross < 100_000)).astype(float)
            - 0.30 * ((startup < 0) & (gross > 500_000)).astype(float)
            + VINTAGE_SHIFT[year]
            + rng.normal(0, 0.55, n)                             # unobserved heterogeneity
        )
        # lender pricing sees only part of the risk => interest rate carries signal, not the answer
        prime = config.PRIME_RATE[year]
        spread = 2.6 + 0.55 * (gross < 50_000) + 0.35 * (term <= 84) + 0.6 * np.tanh(0.55 * (z - VINTAGE_SHIFT[year])) \
            + rng.normal(0, 0.45, n)
        rate = np.where(is504, 4.6 + 0.35 * (prime - 3.25) + rng.normal(0, .35, n),
                        np.where(df["FixedOrVariableInterestInd"] == "V", prime + spread, prime + spread + 0.8))
        df["InitialInterestRate"] = np.round(rate.clip(2.5, 18), 3)
        df["_z"] = z
        df["_year"] = year
        frames.append(df)

    d = pd.concat(frames, ignore_index=True)

    # ---- calibrate intercept so overall 36m default rate ~ target_rate over 2010-2021 vintages
    mask = d["_year"] <= 2021
    lo, hi = -6.0, 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _sigmoid(d.loc[mask, "_z"] + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    pd36 = _sigmoid(d["_z"] + (lo + hi) / 2).to_numpy()
    n = len(d)
    rng2 = np.random.default_rng(seed + 1)

    defaulted36 = rng2.random(n) < pd36
    late_default = (~defaulted36) & (rng2.random(n) < 0.6 * pd36)   # charge-offs AFTER 36 months (label = 0)
    months_to_co = np.clip(np.round(rng2.gamma(3.4, 6.0, n)), 2, 36)
    months_late = rng2.integers(37, 85, n)
    co_months = np.where(defaulted36, months_to_co, np.where(late_default, months_late, np.nan))
    co_date = d["FirstDisbursementDate"] + pd.to_timedelta(co_months * 30.4, unit="D")
    charged_off = co_date.notna() & (co_date <= as_of)

    # paid-in-full among non-charged-off loans
    pif_months = np.minimum(d["TermInMonths"].to_numpy(), rng2.gamma(4, 14, n) + 6)
    pif_date = d["FirstDisbursementDate"] + pd.to_timedelta(pif_months * 30.4, unit="D")
    paid = (~charged_off) & (~(defaulted36 | late_default)) & (pif_date <= as_of) & (rng2.random(n) < 0.7)

    status = np.where(charged_off, "CHGOFF", np.where(paid, "PIF", "EXEMPT"))
    cancelled = rng2.random(n) < 0.04                     # approved but never disbursed
    status = np.where(cancelled, "CANCLD", status)
    committed = (d["ApprovalDate"] > as_of - pd.Timedelta(days=150)) & (rng2.random(n) < .3)
    status = np.where(committed & ~cancelled, "COMMIT", status)
    d["LoanStatus"] = status
    d["ChargeOffDate"] = co_date.where(status == "CHGOFF").dt.normalize()
    d["PaidInFullDate"] = pif_date.where(status == "PIF").dt.normalize()
    ratio = rng2.beta(6, 2.5, n)                          # outstanding at charge-off / gross approval
    d["GrossChargeOffAmount"] = np.where(status == "CHGOFF", np.round(d["GrossApproval"] * ratio, 0), np.nan)
    d.loc[status == "CANCLD", "FirstDisbursementDate"] = pd.NaT
    d["SoldSecMrktInd"] = np.where(rng2.random(n) < 0.18, "Y", "N")
    d["AsOfDate"] = as_of

    # ---- realistic messiness: missing values / unanswered fields that vary by vintage
    m = rng2.random(n)
    d.loc[(d["_year"] <= 2011) & (m < .35), "CollateralInd"] = None
    d.loc[(m > .97), "NaicsCode"] = None
    d.loc[(m < .02), "InitialInterestRate"] = np.nan
    d.loc[(d["_year"] <= 2012) & (m < .30), "BusinessAge"] = "Unanswered"
    d.loc[(rng2.random(n) < .01), "BusinessType"] = None
    d.loc[(rng2.random(n) < .006), "BorrState"] = None

    d = d.drop(columns=["_z", "_year"])
    d = d.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for c in ["ApprovalDate", "FirstDisbursementDate", "ChargeOffDate", "PaidInFullDate", "AsOfDate"]:
        d[c] = pd.to_datetime(d[c]).dt.strftime("%m/%d/%Y")
    d["ApprovalFiscalYear"] = pd.to_datetime(d["ApprovalDate"]).dt.year
    d["BorrName"] = "SYNTHETIC BORROWER " + d.index.astype(str)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-per-year", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--out", default=str(config.DATA_SYNTH / "foia_synthetic.csv"))
    a = ap.parse_args()
    df = generate(a.n_per_year, a.seed)
    out = pd.io.common.stringify_path(a.out)
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} synthetic rows -> {out}")
    print(df["LoanStatus"].value_counts().to_string())


if __name__ == "__main__":
    main()
