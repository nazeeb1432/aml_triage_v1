"""Splits, tuning, feature selection, calibration and the frozen operating threshold.

Time discipline:
  warm-up | train | val | test, split by alert day, never shuffled.
  With respect_label_delay, a train row is used only if its decision was recorded before
  validation starts, and a val row only if decided before test starts: a model trained "on
  day X" cannot see labels that only exist later. Test uses hindsight labels (evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from aml_triage.config import Config
from aml_triage.evaluation.metrics import threshold_at_fpr


def split(table: pd.DataFrame, cfg: Config) -> pd.Series:
    s = cfg.split
    lab = pd.Series("drop", index=table.index)
    d, dec = table["day"], table["decision_day"]
    ignore_delay = not s.respect_label_delay  # plain bool: `~True` would be -2, which is truthy
    lab[(d >= s.warmup_days) & (d < s.val_start_day) & ((dec < s.val_start_day) | ignore_delay)] = "train"
    lab[(d >= s.val_start_day) & (d < s.test_start_day) & ((dec < s.test_start_day) | ignore_delay)] = "val"
    lab[d >= s.test_start_day] = "test"
    lab[d < s.warmup_days] = "warmup"
    return lab


def _objective_metric(y, s) -> float:
    # standardised partial AUC up to FPR 0.3: smooth, and focused on the region where we operate
    return float(roc_auc_score(y, s, max_fpr=0.3))


def tune(Xtr, ytr, Xva, yva, cfg: Config, tune_class_weight: bool = True) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True) if tune_class_weight else 1.0,
        }
        m = lgb.LGBMClassifier(**params, random_state=cfg.seed, verbosity=-1, n_jobs=2)
        m.fit(Xtr, ytr)
        return _objective_metric(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(obj, n_trials=cfg.training.n_trials, timeout=cfg.training.timeout_seconds)
    best = dict(study.best_params)
    best.setdefault("scale_pos_weight", 1.0)
    best["subsample_freq"] = 1
    return best


def select_features(model: lgb.LGBMClassifier, cols: list[str], share: float) -> list[str]:
    """Smallest set of features covering `share` of total gain (Eddin §2.2 uses 90% of the metric)."""
    gain = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    if gain.sum() == 0:
        return cols
    keep = gain.index[: int(np.searchsorted(gain.cumsum() / gain.sum(), share) + 1)]
    return list(keep)


@dataclass
class TrainedModel:
    model: lgb.LGBMClassifier
    calibrator: IsotonicRegression | PlattCalibrator
    features: list[str]
    params: dict
    tau_raw: float             # frozen threshold on the raw score (decisions use this)
    feature_set: str = ""
    augmentation: str = "class_weight"
    fill_nan: bool = False     # augmentation arms train on NaN-filled data, so score the same way
    extra: dict = field(default_factory=dict)

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.features]
        if self.fill_nan:
            Xf = Xf.astype("float64").fillna(-1.0)
        return self.model.predict_proba(Xf)[:, 1]

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability: shown to analysts and used for BTL strata."""
        return self.calibrator.predict(self.raw(X))

    @property
    def tau(self) -> float:
        return float(self.calibrator.predict([self.tau_raw])[0])


class PlattCalibrator:
    """Logistic regression on the logit of the raw score. Smooth and strictly between 0 and 1, so it stays usable when
    validation has few positives (isotonic regression then collapses whole score ranges to exactly 0)."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _z(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw, y) -> PlattCalibrator:
        self.lr.fit(self._z(raw), np.asarray(y).astype(int))
        return self

    def predict(self, raw) -> np.ndarray:
        return self.lr.predict_proba(self._z(raw))[:, 1]


def make_calibrator(raw_va, yva, cfg: Config):
    method = cfg.training.calibration
    if method == "auto":
        method = "isotonic" if int(np.asarray(yva).sum()) >= cfg.training.isotonic_min_positives else "sigmoid"
    if method == "sigmoid":
        return PlattCalibrator().fit(raw_va, yva)
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, yva)


def fit_final(Xtr, ytr, Xva, yva, params: dict, cfg: Config) -> tuple[lgb.LGBMClassifier, IsotonicRegression, float]:
    """Fit, calibrate on validation (isotonic), and freeze the threshold at the target FPR on validation.
    Isotonic output is a step function, so decisions use the raw score threshold; the calibrated
    probability is what people see."""
    m = lgb.LGBMClassifier(**params, random_state=cfg.seed, verbosity=-1, n_jobs=2)
    m.fit(Xtr, ytr)
    raw_va = m.predict_proba(Xva)[:, 1]
    cal = make_calibrator(raw_va, yva, cfg)
    tau_raw = threshold_at_fpr(yva, raw_va, cfg.training.target_fpr)
    return m, cal, tau_raw
