import numpy as np
import pandas as pd
import pytest

from src import features
from tests.conftest import make_loans


def test_leakage_registry_covers_all_outcome_fields():
    for col in ["LoanStatus", "ChargeOffDate", "GrossChargeOffAmount", "PaidInFullDate", "SoldSecMrktInd"]:
        assert col in features.LEAKAGE_EXCLUDED and col in features._FORBIDDEN


def test_features_contain_no_forbidden_columns(labelled):
    X = features.make_features(labelled[0])
    assert not set(X.columns) & features._FORBIDDEN
    assert list(X.columns) == features.FEATURES


@pytest.mark.parametrize("bad", ["ChargeOffDate", "LoanStatus", "default", "approval_year", "GrossChargeOffAmount"])
def test_leakage_guard_raises(bad):
    with pytest.raises(ValueError, match="Leakage guard"):
        features.assert_no_leakage(features.FEATURES + [bad])


def test_features_do_not_change_when_outcome_fields_change():
    """Direct leak test: flipping every post-origination field must not move a single feature value."""
    a = make_loans(LoanStatus="PIF")
    b = make_loans(LoanStatus="CHGOFF", ChargeOffDate=pd.Timestamp("2016-01-01"), GrossChargeOffAmount=90000.0,
                   PaidInFullDate=pd.Timestamp("2017-01-01"), SoldSecMrktInd="Y", AsOfDate=pd.Timestamp("2030-01-01"))
    pd.testing.assert_frame_equal(features.make_features(a), features.make_features(b))


def test_numeric_engineering():
    raw = make_loans(GrossApproval=200000.0, SBAGuaranteedApproval=150000.0, InitialInterestRate=8.25, JobsSupported=10.0,
                     ApprovalDate=pd.Timestamp("2019-05-01"))
    X = features.make_features(raw).iloc[0]
    assert X["guarantee_pct"] == pytest.approx(0.75)
    assert X["log_gross_approval"] == pytest.approx(np.log(200000))
    assert X["rate_spread"] == pytest.approx(8.25 - 5.28)  # prime for 2019
    assert X["jobs_per_100k"] == pytest.approx(5.0)


def test_invalid_amounts_become_nan_not_inf():
    X = features.make_features(make_loans(GrossApproval=0.0)).iloc[0]
    assert np.isnan(X["log_gross_approval"]) and np.isnan(X["guarantee_pct"]) and np.isnan(X["jobs_per_100k"])


def test_guarantee_pct_is_clipped():
    assert features.make_features(make_loans(GrossApproval=100.0, SBAGuaranteedApproval=500.0)).iloc[0]["guarantee_pct"] == 1.0


def test_categoricals_are_normalised_and_lightgbm_safe():
    raw = make_loans(BusinessAge="Startup, Loan Funds will Open Business", BorrState=" tx ", NaicsCode="722511", FranchiseCode="12345")
    X = features.make_features(raw).iloc[0]
    assert X["business_age"] == "Startup_Loan_Funds_will_Open_Business"
    assert X["borrower_state"] == "TX" and X["naics_sector"] == "72" and X["is_franchise"] == "Y" and X["non_profit"] == "N"


def test_all_missing_optional_columns_do_not_crash():
    X = features.make_features(make_loans())  # BorrState/NaicsCode/... all NaN
    assert X.shape == (1, len(features.FEATURES)) and pd.isna(X.iloc[0]["borrower_state"])


def test_year_beyond_prime_table_is_clipped_not_nan():
    X = features.make_features(make_loans(InitialInterestRate=9.0, ApprovalDate=pd.Timestamp("2031-01-01")))
    assert X["rate_spread"].notna().all()


def test_preprocessor_fits_on_train_only_and_handles_unseen_levels(labelled):
    df = labelled[0]
    X = features.make_features(df)
    train, test = X.iloc[:2000], X.iloc[2000:2500].copy()
    prep = features.build_preprocessor(scale=True).fit(train)
    # imputer statistics come from train, not from test
    imp = prep.named_transformers_["num"].named_steps["impute"]
    assert imp.statistics_[0] == pytest.approx(train["log_gross_approval"].median())
    assert imp.statistics_[0] != pytest.approx(test["log_gross_approval"].median())
    test.iloc[0, test.columns.get_loc("borrower_state")] = "ZZ"  # unseen category
    test.iloc[1, test.columns.get_loc("term_months")] = np.nan
    out = prep.transform(test)
    assert not out.isna().any().any() and out.shape[0] == len(test)
    assert list(out.columns) == list(prep.transform(train.head(3)).columns)
