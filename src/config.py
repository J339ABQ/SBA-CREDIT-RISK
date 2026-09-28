"""Central configuration: paths, dates, split boundaries and modelling constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_REAL = DATA_RAW / "real"            # where real SBA FOIA CSVs go (download / manual)
DATA_SYNTH = DATA_RAW / "synthetic"      # where the synthetic stand-in is written
DATA_SAMPLE = ROOT / "data" / "sample"   # small committed sample (for API demo / tests)
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "pd_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42

# ---- Target definition -------------------------------------------------------
# default = 1  <=>  ChargeOffDate exists AND falls within OBS_MONTHS of first disbursement.
OBS_MONTHS = 36
# Date the extract was pulled. A loan is only labelled if it has been observable for
# a full OBS_MONTHS window before this date (otherwise its label is right-censored).
AS_OF_DATE = "2025-06-30"

# ---- Out-of-time split (by approval fiscal-year of the loan) ---------------------
TRAIN_YEARS = (2010, 2016)   # fit + hyper-parameter CV (forward-chaining inside this window)
VALID_YEARS = (2017, 2018)   # calibration + cut-off selection
TEST_YEARS = (2019, 2021)    # untouched final evaluation (includes COVID-era vintages)

# ---- Approximate WSJ prime rate (year average, %), used to de-trend loan rates -----
PRIME_RATE = {
    2010: 3.25, 2011: 3.25, 2012: 3.25, 2013: 3.25, 2014: 3.25, 2015: 3.26,
    2016: 3.51, 2017: 4.10, 2018: 4.91, 2019: 5.28, 2020: 3.54, 2021: 3.25,
    2022: 4.85, 2023: 8.20, 2024: 8.50, 2025: 7.50,
}
