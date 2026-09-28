import numpy as np
import pandas as pd
import pytest

from src import config, data_prep
from tests.conftest import make_loans

AS_OF = "2025-06-30"


def target(**kw):
    out, _ = data_prep.build_target(make_loans(**kw), as_of=AS_OF)
    return out


def test_chargeoff_inside_window_is_default():
    out = target(LoanStatus="CHGOFF", ChargeOffDate=pd.Timestamp("2016-08-01"))  # ~18m after disbursement
    assert out["default"].tolist() == [1]


def test_chargeoff_exactly_at_window_end_is_default_and_one_day_later_is_not():
    end = pd.Timestamp("2015-02-01") + pd.DateOffset(months=36)
    assert target(LoanStatus="CHGOFF", ChargeOffDate=end)["default"].iloc[0] == 1
    late = target(LoanStatus="CHGOFF", ChargeOffDate=end + pd.Timedelta(days=1))
    assert late["default"].iloc[0] == 0  # late charge-offs are labelled 0 by definition


def test_paid_in_full_and_in_force_are_not_default():
    assert target(LoanStatus="PIF", PaidInFullDate=pd.Timestamp("2018-01-01"))["default"].iloc[0] == 0
    assert target(LoanStatus="EXEMPT")["default"].iloc[0] == 0


@pytest.mark.parametrize("status", ["CANCLD", "COMMIT"])
def test_unfunded_statuses_are_dropped(status):
    assert target(LoanStatus=status).empty


def test_right_censored_loans_are_dropped():
    # disbursed 20 months before the as-of date: full 36m window not observed -> label unknowable
    assert target(FirstDisbursementDate=pd.Timestamp("2023-10-30"), ApprovalDate=pd.Timestamp("2023-10-01")).empty


def test_missing_disbursement_date_is_dropped():
    assert target(FirstDisbursementDate=pd.NaT).empty


def test_stats_are_consistent(labelled):
    df, st = labelled
    assert st["raw_rows"] >= st["after_status_filter"] >= st["after_disbursement_filter"] >= st["after_censoring_filter"] == len(df)
    assert set(df["default"].unique()) <= {0, 1}
    assert 0.02 < st["default_rate"] < 0.2


def test_all_labelled_loans_have_full_window(labelled):
    df, _ = labelled
    assert (df["FirstDisbursementDate"] + pd.DateOffset(months=config.OBS_MONTHS) <= pd.Timestamp(config.AS_OF_DATE)).all()


def test_split_is_disjoint_ordered_and_out_of_time(labelled):
    tr, va, te = data_prep.split_by_vintage(labelled[0])
    assert tr["approval_year"].max() < va["approval_year"].min() <= va["approval_year"].max() < te["approval_year"].min()
    assert len(set(tr.index) & set(va.index)) == 0 and len(set(va.index) & set(te.index)) == 0
    assert min(len(tr), len(va), len(te)) > 100


def test_header_aliases_and_missing_optional_columns_are_harmonised(tmp_path):
    p = tmp_path / "foia-504-sample.csv"  # '504' in name -> program hint when Program column absent
    pd.DataFrame({"terminmonths": [240], "GrossApproval": [500000], "LoanStatus": ["PIF"], "paidinfulldate": ["01/02/2019"],
                  "ApprovalDate": ["01/01/2015"], "FirstDisbursementDate": ["02/01/2015"]}).to_csv(p, index=False)
    df = data_prep.load_raw(p)
    assert list(df.columns) == data_prep.CANONICAL
    assert df.loc[0, "TermInMonths"] == 240 and df.loc[0, "Program"] == "504"
    assert pd.isna(df.loc[0, "ChargeOffDate"]) and df.loc[0, "PaidInFullDate"] == pd.Timestamp("2019-01-02")


def test_real_files_take_precedence_over_synthetic(tmp_path, monkeypatch):
    real, synth = tmp_path / "real", tmp_path / "synth"
    real.mkdir(); synth.mkdir()
    (real / "a.csv").write_text("x\n1\n"); (synth / "b.csv").write_text("x\n1\n")
    monkeypatch.setattr(config, "DATA_REAL", real); monkeypatch.setattr(config, "DATA_SYNTH", synth)
    assert [p.name for p in data_prep.find_raw_files()] == ["a.csv"]
    (real / "a.csv").unlink()
    assert [p.name for p in data_prep.find_raw_files()] == ["b.csv"]


def test_load_raw_without_any_data_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_REAL", tmp_path / "nope"); monkeypatch.setattr(config, "DATA_SYNTH", tmp_path / "nope2")
    with pytest.raises(FileNotFoundError, match="data_download"):
        data_prep.load_raw()
