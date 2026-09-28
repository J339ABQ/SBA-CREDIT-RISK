"""Discrimination, calibration, drift (PSI), importance and plotting helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from . import config

PALETTE = {"lr": "#7a8797", "gbm": "#1f6feb", "cal": "#0f9d75", "bad": "#d1495b", "accent": "#e8a33d", "ink": "#22303c"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelcolor": PALETTE["ink"], "font.size": 10, "legend.frameon": False,
    })


def save_fig(fig, name: str) -> str:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return str(path.relative_to(config.ROOT))


# ------------------------------------------------------------------ discrimination -------------
def ks_statistic(y, p) -> float:
    """Max separation between the CDFs of scores for defaulters and non-defaulters."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def discrimination_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "default_rate": float(y.mean()), "mean_pd": float(p.mean()),
            "auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, p),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}


def bootstrap_auc_ci(y, p, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------- calibration ----------
def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index(drop=True)
    return t


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float(np.sum(t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()) / t["n"].sum())


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Regress outcome on logit(p): slope 1 / intercept 0 = perfect. Slope<1 => over-dispersed (overconfident) scores."""
    from sklearn.linear_model import LogisticRegression
    pp = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lg = np.log(pp / (1 - pp)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg, np.asarray(y))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


# ---------------------------------------------------------------------------- drift -----------
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of a score/feature between two samples (rule of thumb: <0.1 stable, >0.25 shifted)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected[~np.isnan(expected)], edges)[0] / max((~np.isnan(expected)).sum(), 1)
    a = np.histogram(actual[~np.isnan(actual)], edges)[0] / max((~np.isnan(actual)).sum(), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ------------------------------------------------------------------------ importance ---------
def permutation_importance_auc(pipeline, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 0, max_rows: int = 20000) -> pd.DataFrame:
    """Permute each ORIGINAL feature (before encoding) and record the AUC drop - one bar per business variable."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    y = np.asarray(y)
    base = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - roc_auc_score(y, pipeline.predict_proba(Xp)[:, 1]))
        rows.append((col, np.mean(drops), np.std(drops)))
    return pd.DataFrame(rows, columns=["feature", "auc_drop", "std"]).sort_values("auc_drop", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- plots ----------
def plot_roc(curves: dict[str, tuple], title: str = "ROC curve (out-of-time test)"):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    colors = [PALETTE["lr"], PALETTE["gbm"], PALETTE["accent"]]
    for (name, (y, p)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    return fig


def plot_ks(y, p, title: str = "KS: score separation (test)"):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    ys = y[order]
    x = np.arange(1, len(y) + 1) / len(y)
    bad, good = np.cumsum(ys) / ys.sum(), np.cumsum(1 - ys) / (1 - ys).sum()
    k = np.argmax(np.abs(bad - good))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(x, good, color=PALETTE["lr"], lw=2, label="Non-defaulters (cum.)")
    ax.plot(x, bad, color=PALETTE["bad"], lw=2, label="Defaulters (cum.)")
    ax.vlines(x[k], bad[k], good[k], color=PALETTE["ink"], lw=2)
    ax.annotate(f"KS = {abs(bad[k] - good[k]):.3f}", (x[k], (bad[k] + good[k]) / 2), xytext=(8, 0), textcoords="offset points")
    ax.set(xlabel="Share of loans, ranked low -> high PD", ylabel="Cumulative share", title=title)
    ax.legend(loc="upper left")
    return fig


def plot_calibration(curves: dict[str, tuple], title: str, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    colors = [PALETTE["lr"], PALETTE["bad"], PALETTE["cal"], PALETTE["gbm"]]
    hi = 0.0
    for (name, (y, p)), c in zip(curves.items(), colors):
        t = calibration_table(y, p, n_bins)
        ax.plot(t["mean_pred"], t["obs_rate"], "o-", color=c, lw=2, ms=4, label=f"{name} (ECE {expected_calibration_error(y, p, n_bins):.3f})")
        hi = max(hi, t["mean_pred"].max(), t["obs_rate"].max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], "--", color="#aaa", lw=1, label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD (decile)", ylabel="Observed default rate", title=title)
    ax.legend(loc="upper left")
    return fig


def plot_decile_lift(y, p, title: str = "Default rate by predicted-PD decile (test)"):
    t = calibration_table(y, p, 10)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(np.arange(1, len(t) + 1), t["obs_rate"] * 100, color=PALETTE["gbm"], alpha=.9, label="Observed")
    ax.plot(np.arange(1, len(t) + 1), t["mean_pred"] * 100, "o-", color=PALETTE["accent"], label="Mean predicted")
    ax.axhline(np.mean(y) * 100, color="#888", ls="--", lw=1)
    ax.set(xlabel="PD decile (1 = safest)", ylabel="Default rate (%)", title=title)
    ax.legend(loc="upper left")
    return fig
