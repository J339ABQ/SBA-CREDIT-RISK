import numpy as np
import pandas as pd
import pytest

from src import business as biz
from src import evaluation as ev
from src import modeling


# ------------------------------------------------------------------ evaluation
def test_ks_perfect_and_random():
    y = np.array([0] * 500 + [1] * 500)
    assert ev.ks_statistic(y, np.r_[np.zeros(500), np.ones(500)]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert ev.ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.06


def test_gini_is_2auc_minus_1_and_metrics_keys():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.3 + 0.2 * y + rng.normal(0, .2, 2000), 0.01, 0.99)
    m = ev.discrimination_metrics(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1) and 0.5 < m["auc"] < 1
    assert {"auc", "gini", "ks", "brier", "log_loss", "mean_pd", "default_rate", "n"} <= set(m)


def test_psi_zero_for_identical_and_large_for_shifted():
    x = np.random.default_rng(0).normal(size=20000)
    assert ev.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert ev.psi(x, x + 2.0) > 0.25


def test_calibration_table_and_ece_for_perfectly_calibrated_scores():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.3, 60000)
    y = (rng.random(60000) < p).astype(int)
    assert ev.expected_calibration_error(y, p) < 0.01
    assert ev.expected_calibration_error(y, p * 0.5) > 0.03
    slope, icpt = ev.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.1) and icpt == pytest.approx(0, abs=0.1)


# ------------------------------------------------------------------- modeling
def test_forward_chaining_never_trains_on_the_future():
    years = np.repeat(np.arange(2010, 2017), 50)
    folds = list(modeling.forward_chaining_splits(years))
    assert len(folds) == 4
    for tr, va in folds:
        assert years[tr].max() < years[va].min() and len(set(years[va])) == 1


@pytest.mark.parametrize("cls", [modeling.PlattCalibrator, modeling.IsotonicCalibrator])
def test_calibrators_fix_a_biased_score_and_stay_in_bounds(cls):
    rng = np.random.default_rng(3)
    true = rng.uniform(0.02, 0.25, 30000)
    y = (rng.random(30000) < true).astype(int)
    biased = np.clip(true * 0.5, 1e-3, 0.99)  # systematically under-predicts by half
    cal = cls().fit(biased, y)
    out = cal.predict(biased)
    assert out.min() >= 0 and out.max() <= 1
    assert abs(out.mean() - y.mean()) < 0.01 < abs(biased.mean() - y.mean())
    order = np.argsort(biased)
    assert (np.diff(out[order]) >= -1e-9).all()  # monotone: never changes ranking


def test_choose_calibrator_returns_fitted_calibrator():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.3, 8000); y = (rng.random(8000) < p * 1.3).astype(int); yrs = rng.choice([2017, 2018], 8000)
    cal, scores = modeling.choose_calibrator(p, y, yrs)
    assert cal.name in scores and set(scores) == {"platt", "isotonic"} and 0 < cal.predict(np.array([0.1]))[0] < 1


def test_pipelines_train_end_to_end_and_beat_chance(labelled):
    from src import data_prep, features
    from sklearn.metrics import roc_auc_score
    tr, _, te = data_prep.split_by_vintage(labelled[0])
    for pipe in (modeling.baseline_pipeline(), modeling.gbm_pipeline(n_estimators=60, n_jobs=2)):
        pipe.fit(features.make_features(tr), tr["default"])
        auc = roc_auc_score(te["default"], pipe.predict_proba(features.make_features(te))[:, 1])
        assert auc > 0.6


# -------------------------------------------------------------------- business
ECON = biz.Economics(lgd=0.5, ead_factor=0.8, net_margin=0.02, avg_balance_factor=0.9, horizon_years=3.0)


def test_expected_loss_formula_and_breakeven():
    assert ECON.el_rate(0.10) == pytest.approx(0.10 * 0.5 * 0.8)
    assert ECON.revenue_rate == pytest.approx(0.02 * 0.9 * 3)
    assert ECON.breakeven_pd == pytest.approx(ECON.revenue_rate / (0.5 * 0.8))
    # at the break-even PD, expected profit per $ is zero
    assert ECON.revenue_rate - ECON.el_rate(ECON.breakeven_pd) == pytest.approx(0)


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.3, n)
    df = pd.DataFrame({"GrossApproval": rng.uniform(5e4, 5e5, n), "default": (rng.random(n) < p).astype(int)})
    df["GrossChargeOffAmount"] = np.where(df["default"] == 1, df["GrossApproval"] * 0.7, np.nan)
    return df, p


def test_realised_loss_uses_actual_chargeoff_amount_and_only_for_defaults():
    df, _ = _toy()
    loss = biz.realised_loss(df, ECON)
    assert (loss[df["default"] == 0] == 0).all()
    d = df[df["default"] == 1].iloc[0]
    assert loss[df["default"].to_numpy() == 1][0] == pytest.approx(d["GrossChargeOffAmount"] * ECON.lgd)


def test_cutoff_curve_is_monotone_and_endpoint_equals_approve_all():
    df, p = _toy()
    c = biz.cutoff_curve(df, p, ECON)
    assert (np.diff(c["approval_rate"]) >= 0).all() and (np.diff(c["approved_dollars"]) >= 0).all()
    assert (np.diff(c["expected_loss"]) >= -1e-6).all()
    last = c.iloc[-1]
    assert last["approval_rate"] == 1.0 and last["default_rate"] == pytest.approx(df["default"].mean())
    # a well-ranking score gives a lower default rate at tight cut-offs
    assert c.iloc[5]["default_rate"] < c.iloc[-1]["default_rate"]


def test_recommend_cutoff_is_tightest_within_tolerance_of_max_profit():
    df, p = _toy()
    c = biz.cutoff_curve(df, p, ECON)
    r = biz.recommend_cutoff(c, tolerance=0.01)
    assert r["expected_profit"] >= 0.99 * c["expected_profit"].max()
    assert r["cutoff"] <= c.loc[c["expected_profit"].idxmax(), "cutoff"]
    assert biz.recommend_cutoff(c, tolerance=0.0)["expected_profit"] == c["expected_profit"].max()


def test_compare_strategies_rows_and_matched_volume():
    df, p = _toy()
    decline = np.random.default_rng(1).random(len(df)) < 0.2
    out = biz.compare_strategies(df, p, decline, ECON, model_cutoff=0.12)
    assert out.loc[0, "approval_rate"] == 1.0 and out.loc[0, "loss_rate_vs_approve_all"] == 0
    assert out.loc[2, "approval_rate"] == pytest.approx(out.loc[1, "approval_rate"], abs=1e-3)
    assert out.loc[2, "default_rate"] < out.loc[1, "default_rate"]  # ranking by PD beats random declines


def test_ead_estimate_and_fallback():
    df, _ = _toy()
    assert biz.estimate_ead_factor(df) == pytest.approx(0.7)
    assert biz.estimate_ead_factor(df.head(10)) == 0.75


def test_pricing_table_required_spread_increases_with_risk():
    df, p = _toy()
    t = biz.pricing_table(df, p, ECON, n_bands=5)
    assert len(t) == 5 and t["required_spread_pct"].is_monotonic_increasing
