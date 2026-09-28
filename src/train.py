"""End-to-end training: raw data -> tuned models -> calibration -> cut-off -> saved artefacts.

    python -m src.train            # full run (about 3-6 minutes on 4 cores)
    python -m src.train --quick    # fewer search iterations, for smoke tests
The notebooks call the same stage functions so that notebook results == CLI results.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import business, config, data_prep, evaluation, features, modeling

log = logging.getLogger("train")


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    stats: dict

    def X(self, name):
        return features.make_features(getattr(self, name))

    def y(self, name):
        return getattr(self, name)[features.TARGET]


def prepare(path=None) -> Splits:
    df, stats = data_prep.load_modeling_table(path)
    tr, va, te = data_prep.split_by_vintage(df)
    for d in (tr, va, te):
        d.reset_index(drop=True, inplace=True)
    return Splits(tr, va, te, stats)


def data_source() -> str:
    files = data_prep.find_raw_files()
    return "synthetic" if files and "synthetic" in str(files[0]) else "real"


def train(n_iter: int = 20, path=None, splits: Splits | None = None) -> dict:
    """Run every modelling stage and return all intermediate objects/metrics."""
    t0 = time.time()
    s = splits or prepare(path)
    Xtr, ytr, yrs = s.X("train"), s.y("train"), s.train["approval_year"]

    base_search = modeling.tune_baseline(Xtr, ytr, yrs)
    gbm_search = modeling.tune_gbm(Xtr, ytr, yrs, n_iter=n_iter)
    base, gbm = base_search.best_estimator_, gbm_search.best_estimator_

    raw = {name: {"lr": base.predict_proba(s.X(name))[:, 1], "gbm": gbm.predict_proba(s.X(name))[:, 1]}
           for name in ("valid", "test")}
    # ---- calibration on the validation vintages (never on train, never on test)
    cal, cal_scores = modeling.choose_calibrator(raw["valid"]["gbm"], s.y("valid").to_numpy(), s.valid["approval_year"].to_numpy())
    cal_valid, cal_test = cal.predict(raw["valid"]["gbm"]), cal.predict(raw["test"]["gbm"])

    # ---- business layer: assumptions, cut-off chosen on VALID, evaluated on TEST
    econ = business.Economics(ead_factor=business.estimate_ead_factor(s.train))
    curve_valid = business.cutoff_curve(s.valid, cal_valid, econ)
    rec = business.recommend_cutoff(curve_valid)
    cutoff = float(rec["cutoff"])
    band_edges = [float(x) for x in np.quantile(cal_valid, [0.2, 0.4, 0.6, 0.8])]

    metrics = {
        "data_source": data_source(), "data_stats": s.stats, "econ": econ.to_dict(), "cutoff": cutoff,
        "cv_auc": {"logistic": float(base_search.best_score_), "lightgbm": float(gbm_search.best_score_)},
        "best_params": {"logistic": base_search.best_params_,
                        "lightgbm": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in gbm_search.best_params_.items()}},
        "calibration_method": cal.name, "calibration_cv_brier": cal_scores,
        "splits": {k: {"years": list(getattr(config, f"{k.upper()}_YEARS")), "n": len(getattr(s, k)),
                       "default_rate": float(getattr(s, k)["default"].mean())} for k in ("train", "valid", "test")},
    }
    for name in ("valid", "test"):
        y = s.y(name).to_numpy()
        metrics[name] = {
            "logistic": evaluation.discrimination_metrics(y, raw[name]["lr"]),
            "lightgbm_raw": evaluation.discrimination_metrics(y, raw[name]["gbm"]),
            "lightgbm_calibrated": evaluation.discrimination_metrics(y, cal_valid if name == "valid" else cal_test),
        }
    metrics["test"]["auc_ci95_lightgbm"] = evaluation.bootstrap_auc_ci(s.y("test"), raw["test"]["gbm"])
    metrics["test"]["ece_raw"] = evaluation.expected_calibration_error(s.y("test"), raw["test"]["gbm"])
    metrics["test"]["ece_calibrated"] = evaluation.expected_calibration_error(s.y("test"), cal_test)
    metrics["test"]["calibration_slope_intercept"] = evaluation.calibration_slope_intercept(s.y("test"), cal_test)
    metrics["test"]["psi_score_train_vs_test"] = evaluation.psi(gbm.predict_proba(Xtr)[:, 1], raw["test"]["gbm"])
    yr = s.test.assign(p=cal_test).groupby("approval_year").agg(n=("default", "size"), observed=("default", "mean"), predicted=("p", "mean"))
    metrics["test"]["by_vintage"] = yr.reset_index().to_dict("records")

    model = modeling.PDModel(
        pipeline=gbm, calibrator=cal, cutoff=cutoff, lgd=econ.lgd, ead_factor=econ.ead_factor,
        meta={"trained_on": metrics["data_source"], "train_years": list(config.TRAIN_YEARS),
              "calibration_years": list(config.VALID_YEARS), "obs_months": config.OBS_MONTHS,
              "calibrator": cal.name, "band_edges": band_edges, "test_auc": metrics["test"]["lightgbm_calibrated"]["auc"],
              "features": features.FEATURES})
    log.info("training done in %.0fs", time.time() - t0)
    return dict(splits=s, base=base, gbm=gbm, base_search=base_search, gbm_search=gbm_search, raw=raw,
                cal=cal, cal_valid=cal_valid, cal_test=cal_test, econ=econ, curve_valid=curve_valid,
                rec=rec, cutoff=cutoff, metrics=metrics, model=model)


def save(res: dict) -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(res["model"], config.MODEL_PATH, compress=3)
    config.METRICS_PATH.write_text(json.dumps(res["metrics"], indent=2, default=float))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    res = train(n_iter=5 if a.quick else 20, path=a.data)
    save(res)
    m = res["metrics"]
    print(f"data={m['data_source']}  cutoff={m['cutoff']:.3%}  test AUC (GBM)={m['test']['lightgbm_raw']['auc']:.3f}  "
          f"(LR {m['test']['logistic']['auc']:.3f})  -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
