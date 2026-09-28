"""Models, time-aware CV, hyper-parameter search and probability calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor, make_features


# ---------------------------------------------------------------------------- CV --------------
def forward_chaining_splits(years: pd.Series | np.ndarray, min_train_years: int = 3):
    """Expanding-window CV by approval year: train on years < k, validate on year k.

    A random K-fold would put 2015 loans in the training fold of a 2013 validation fold - i.e. the
    tuning would reward models that exploit *future* economic conditions. This keeps tuning honest.
    """
    years = np.asarray(years)
    uniq = np.sort(np.unique(years))
    for k in uniq[min_train_years:]:
        yield np.where(years < k)[0], np.where(years == k)[0]


# ------------------------------------------------------------------------- models -------------
def baseline_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(scale=True)),
        ("clf", LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
    ])


def gbm_pipeline(**params) -> Pipeline:
    defaults = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=config.RANDOM_STATE, n_jobs=4, verbose=-1)
    defaults.update(params)
    return Pipeline([("prep", build_preprocessor(scale=False)), ("clf", LGBMClassifier(**defaults))])


def tune_baseline(X: pd.DataFrame, y: pd.Series, years: pd.Series) -> GridSearchCV:
    cv = list(forward_chaining_splits(years))
    gs = GridSearchCV(baseline_pipeline(), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
                      scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
    return gs.fit(X, y)


def tune_gbm(X: pd.DataFrame, y: pd.Series, years: pd.Series, n_iter: int = 20) -> RandomizedSearchCV:
    cv = list(forward_chaining_splits(years))
    space = {
        "clf__n_estimators": randint(150, 600),
        "clf__learning_rate": loguniform(0.01, 0.1),
        "clf__num_leaves": randint(8, 48),
        "clf__min_child_samples": randint(30, 300),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(0.1, 30),
    }
    rs = RandomizedSearchCV(gbm_pipeline(), space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                            n_jobs=1, random_state=config.RANDOM_STATE, refit=True)
    return rs.fit(X, y)


# --------------------------------------------------------------------- calibration -------------
class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(np.asarray(p))


class PlattCalibrator:
    """Logistic regression on the logit of the raw score (a.k.a. Platt scaling)."""
    name = "platt"

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(self._logit(p))[:, 1]


def choose_calibrator(p: np.ndarray, y: np.ndarray, years: np.ndarray):
    """Pick Platt vs isotonic by cross-fitting across *vintages* inside the validation window."""
    scores = {}
    uniq = np.unique(years)
    for cls in (PlattCalibrator, IsotonicCalibrator):
        briers = []
        for k in uniq:
            tr, te = years != k, years == k
            cal = cls().fit(p[tr], y[tr])
            briers.append(brier_score_loss(y[te], cal.predict(p[te])))
        scores[cls.name] = float(np.mean(briers))
    best = min(scores, key=scores.get)
    return (PlattCalibrator if best == "platt" else IsotonicCalibrator)().fit(p, y), scores


# ------------------------------------------------------------------- deployable model ---------
@dataclass
class PDModel:
    """Everything the API needs: fitted pipeline + calibrator + policy cut-off + metadata."""
    pipeline: Pipeline
    calibrator: object
    cutoff: float
    lgd: float = 0.45
    ead_factor: float = 0.75
    meta: dict = field(default_factory=dict)

    def raw_pd(self, raw_df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(make_features(raw_df))[:, 1]

    def calibrated_pd(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.raw_pd(raw_df)
        return raw, np.clip(self.calibrator.predict(raw), 1e-4, 1 - 1e-4)
