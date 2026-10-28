"""Minority-class augmentation for the TRAINING split only.

What a GAN can and cannot do here:
  * It learns the joint distribution of the feature vectors it is shown and samples more
    rows that look like them. It cannot invent laundering behaviour that isn't in the
    training data, and it knows nothing about the graph the features came from.
  * So it is a (fancier) oversampler. Whether it beats class weights or SMOTE is an
    empirical question, answered by the ablation on real held-out alerts.

Hard rules (enforced by `augment` and by tests):
  1. Fit only on training rows. Validation and test are never augmented or seen.
  2. Synthetic rows are flagged so they can be excluded from every metric.
  3. Fidelity is reported (a classifier tries to tell real from synthetic positives).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENTINEL = -1.0  # NaN carries meaning here (e.g. "no walk reached a suspicious account"); keep it distinct


def fill_for_synthesis(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype("float64").fillna(SENTINEL)


def _n_needed(n_pos: int, n_neg: int, share: float) -> int:
    target_pos = share * n_neg / (1 - share)
    return max(0, int(round(target_pos - n_pos)))


def smote(X: pd.DataFrame, y: pd.Series, share: float, seed: int) -> pd.DataFrame:
    from imblearn.over_sampling import SMOTE

    n_pos, n_neg = int(y.sum()), int((~y.astype(bool)).sum())
    if _n_needed(n_pos, n_neg, share) == 0:
        return X.iloc[:0]
    sm = SMOTE(sampling_strategy=share / (1 - share), k_neighbors=min(5, n_pos - 1), random_state=seed)
    Xr, _ = sm.fit_resample(X, y.astype(int))
    return Xr.iloc[len(X):].reset_index(drop=True)


def ctgan(X: pd.DataFrame, y: pd.Series, share: float, seed: int, epochs: int, batch_size: int,
          negatives_sample: int) -> pd.DataFrame:
    """Conditional tabular GAN (Xu et al., NeurIPS 2019). Fit on all training positives plus a sample of
    negatives with the label as a conditioning column, then sample rows conditioned on label=1."""
    import torch
    from ctgan import CTGAN

    torch.manual_seed(seed)
    np.random.seed(seed)
    yb = y.astype(bool).to_numpy()
    n_need = _n_needed(int(yb.sum()), int((~yb).sum()), share)
    if n_need == 0:
        return X.iloc[:0]
    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(~yb)
    neg_idx = rng.choice(neg_idx, size=min(negatives_sample, len(neg_idx)), replace=False)
    fit_idx = np.concatenate([np.flatnonzero(yb), neg_idx])
    data = X.iloc[fit_idx].copy()
    data["__label__"] = yb[fit_idx].astype(int).astype(str)
    discrete = [c for c in X.columns if X[c].nunique() <= 2] + ["__label__"]
    for c in discrete[:-1]:
        data[c] = data[c].astype(str)
    import warnings

    model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False, cuda=False)
    model.set_random_state(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, discrete_columns=discrete)
    synth = model.sample(n_need, condition_column="__label__", condition_value="1")
    synth = synth.drop(columns="__label__")
    for c in discrete[:-1]:
        synth[c] = synth[c].astype(float)
    # keep samples inside the range seen in real positives; round integer-valued columns
    pos = X.iloc[np.flatnonzero(yb)]
    for c in X.columns:
        lo, hi = X[c].min(), X[c].max()
        synth[c] = synth[c].astype(float).clip(lo, hi)
        if np.allclose(pos[c].dropna() % 1, 0):
            synth[c] = synth[c].round()
    return synth[X.columns].reset_index(drop=True)


def augment(X_train: pd.DataFrame, y_train: pd.Series, method: str, share: float, seed: int,
            ctgan_cfg=None) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return (X, y, is_synthetic). X is NaN-filled for every method so arms are comparable."""
    if "split" in X_train.columns and set(X_train["split"].unique()) != {"train"}:
        raise ValueError("augment() must only ever receive training rows")
    X = fill_for_synthesis(X_train)
    y = y_train.astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if method in ("none", "class_weight"):
        synth = X.iloc[:0]
    elif method == "smote":
        synth = smote(X, y, share, seed)
    elif method == "ctgan":
        synth = ctgan(X, y, share, seed, ctgan_cfg.epochs, ctgan_cfg.batch_size, ctgan_cfg.negatives_sample)
    else:
        raise ValueError(f"unknown augmentation method {method}")
    X_out = pd.concat([X, synth], ignore_index=True)
    y_out = pd.concat([y, pd.Series(np.ones(len(synth), dtype=int))], ignore_index=True)
    flag = np.r_[np.zeros(len(X), bool), np.ones(len(synth), bool)]
    return X_out, y_out, flag


def fidelity_auc(real_pos: pd.DataFrame, synth_pos: pd.DataFrame, seed: int) -> float:
    """5-fold AUC of a classifier separating real from synthetic positives. 0.5 = indistinguishable."""
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    if len(synth_pos) < 10 or len(real_pos) < 10:
        return float("nan")
    n = min(len(real_pos), len(synth_pos))
    rng = np.random.default_rng(seed)
    a = real_pos.sample(n, random_state=seed)
    b = synth_pos.iloc[rng.choice(len(synth_pos), size=n, replace=False)]
    X = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbosity=-1, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())
