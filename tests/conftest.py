import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_prep, synthetic  # noqa: E402


@pytest.fixture(scope="session")
def raw_small() -> pd.DataFrame:
    """~7k synthetic FOIA-schema rows (dates as strings, like the CSV) - fast, deterministic."""
    return synthetic.generate(n_per_year=500, seed=7)


@pytest.fixture(scope="session")
def raw_loaded(raw_small, tmp_path_factory) -> pd.DataFrame:
    p = tmp_path_factory.mktemp("raw") / "foia_test.csv"
    raw_small.to_csv(p, index=False)
    return data_prep.load_raw(p)


@pytest.fixture(scope="session")
def labelled(raw_loaded):
    return data_prep.build_target(raw_loaded)


def make_loans(**overrides) -> pd.DataFrame:
    """One-row canonical frame builder for target-definition edge cases."""
    base = {c: np.nan for c in data_prep.CANONICAL}
    base.update(Program="7A", ApprovalDate=pd.Timestamp("2015-01-10"), FirstDisbursementDate=pd.Timestamp("2015-02-01"),
               GrossApproval=100000.0, SBAGuaranteedApproval=75000.0, TermInMonths=120, LoanStatus="PIF")
    base.update(overrides)
    return pd.DataFrame([base])
