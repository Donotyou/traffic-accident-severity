"""Decision-rule tuning for imbalanced multi-class problems.

A model outputs class probabilities; the default decision (argmax) is not necessarily the
one that maximises macro-F1 when classes are imbalanced. We learn one multiplicative weight
per class on OUT-OF-FOLD training predictions (never on the test set) and predict
argmax(probability * weight). Weights < 1 make the model more conservative about a class,
weights > 1 make it more eager to predict it.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import f1_score


def apply_weights(proba, weights) -> np.ndarray:
    return (np.asarray(proba) * np.asarray(weights, dtype=float)).argmax(axis=1)


def tune_class_weights(y_true, proba, grid_points: int = 17, lo: float = 0.3, hi: float = 6.0):
    """Grid-search weights for classes 1..k-1 (class 0 fixed at 1).

    Returns (weights, score_with_default_argmax, score_with_tuned_weights).
    The grid always contains 1.0, so the tuned score is never below the default one.
    """
    proba = np.asarray(proba)
    k = proba.shape[1]
    score = lambda w: f1_score(y_true, apply_weights(proba, w), average="macro", zero_division=0)
    grid = np.unique(np.round(np.concatenate([np.geomspace(lo, hi, grid_points), [1.0]]), 4))
    base = score(np.ones(k))
    best_w, best_s = np.ones(k), base
    for combo in itertools.product(grid, repeat=k - 1):
        w = np.array([1.0, *combo])
        s = score(w)
        if s > best_s + 1e-12:
            best_w, best_s = w, s
    return best_w, float(base), float(best_s)
