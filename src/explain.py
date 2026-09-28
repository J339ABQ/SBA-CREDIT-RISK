"""Explainability helpers shared by the notebooks and the API (per-loan 'reason codes')."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def original_feature(encoded_name: str) -> str:
    """Map an encoded column (e.g. 'naics_sector_72', 'missingindicator_interest_rate') back to its source feature."""
    name = encoded_name.replace("missingindicator_", "")
    for f in sorted(FEATURES, key=len, reverse=True):
        if name == f or name.startswith(f + "_"):
            return f
    return name


def group_contributions(contrib: np.ndarray, columns) -> pd.DataFrame:
    """Sum encoded-column contributions into one column per original feature."""
    df = pd.DataFrame(contrib, columns=list(columns))
    return df.T.groupby(original_feature).sum().T


def loan_contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-loan, per-feature contribution to the raw log-odds (LightGBM exact tree-SHAP, `pred_contrib`).

    Returns a frame (rows = loans, cols = original features) plus a 'bias' column (the base log-odds).
    """
    Xt = pipeline.named_steps["prep"].transform(X)
    booster = pipeline.named_steps["clf"].booster_
    c = booster.predict(Xt, pred_contrib=True)
    out = group_contributions(c[:, :-1], Xt.columns)
    out["bias"] = c[:, -1]
    return out


def describe_value(feature: str, X_row: pd.Series) -> str:
    v = X_row[feature]
    if feature == "log_gross_approval" and pd.notna(v):
        return f"loan size ${np.exp(v):,.0f}"
    if feature in NUMERIC_FEATURES:
        return f"{feature}={v:.3g}" if pd.notna(v) else f"{feature}=missing"
    return f"{feature}={v}"
