"""Inference wrapper shared by the API and tests: raw application rows -> PD, decision, drivers."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config, explain, features
from .modeling import PDModel  # noqa: F401  (needed so joblib can unpickle the artefact)


def load_model(path: str | Path | None = None) -> PDModel:
    path = Path(path or os.environ.get("MODEL_PATH", config.MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def score(model: PDModel, raw_rows: pd.DataFrame, n_drivers: int = 3) -> list[dict]:
    """Score a frame of canonical raw rows. Returns one dict per row."""
    raw_pd, cal_pd = model.calibrated_pd(raw_rows)
    X = features.make_features(raw_rows)
    contrib = explain.loan_contributions(model.pipeline, X).drop(columns="bias")
    edges = model.meta.get("band_edges", [0.03, 0.05, 0.07, 0.10])
    gross = pd.to_numeric(raw_rows["GrossApproval"]).to_numpy()
    out = []
    for i in range(len(raw_rows)):
        c = contrib.iloc[i]
        top = c.reindex(c.abs().sort_values(ascending=False).index)[:n_drivers]
        el_rate = float(cal_pd[i] * model.lgd * model.ead_factor)
        out.append({
            "pd_raw": float(raw_pd[i]), "pd_calibrated": float(cal_pd[i]),
            "decision": "APPROVE" if cal_pd[i] <= model.cutoff else "DECLINE",
            "cutoff": float(model.cutoff), "risk_band": int(1 + np.searchsorted(edges, cal_pd[i], side="right")),
            "expected_loss_rate": el_rate, "expected_loss_usd": float(el_rate * gross[i]),
            "top_drivers": [{"feature": f, "value": explain.describe_value(f, X.iloc[i]).split("=", 1)[-1],
                             "effect": "raises risk" if v > 0 else "lowers risk", "log_odds_contribution": float(v)}
                            for f, v in top.items()],
        })
    return out
