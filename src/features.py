"""Feature engineering as a single sklearn ColumnTransformer.

Everything (imputation, time encoding, one-hot) lives INSIDE the pipeline, so:
  * it is fitted on training folds only -> no leakage during cross-validation;
  * the exact same transformation is applied at inference time in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _to_hour(value) -> float:
    """'17:02:00' -> 17.03 ; unparsable -> NaN."""
    try:
        parts = str(value).strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        s = int(parts[2]) if len(parts) > 2 else 0
        if not 0 <= h < 24:
            return np.nan
        return h + m / 60 + s / 3600
    except (ValueError, IndexError):
        return np.nan


class CyclicalTime(BaseEstimator, TransformerMixin):
    """Turn a 'HH:MM:SS' column into (sin, cos) of the hour of day + a night flag."""

    def fit(self, X, y=None):
        hours = self._hours(X)
        self.median_hour_ = float(np.nanmedian(hours)) if np.isfinite(hours).any() else 12.0
        return self

    @staticmethod
    def _hours(X) -> np.ndarray:
        s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        return s.map(_to_hour).to_numpy(dtype=float)

    def transform(self, X):
        h = self._hours(X)
        h = np.where(np.isnan(h), self.median_hour_, h)
        angle = 2 * np.pi * h / 24
        night = ((h >= 20) | (h < 6)).astype(float)
        return np.column_stack([np.sin(angle), np.cos(angle), night])

    def get_feature_names_out(self, input_features=None):
        return np.array(["time_sin", "time_cos", "is_night"])


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [
        c for c in X.columns
        if c not in config.NUMERIC_COLS and c != config.TIME_COL
    ]


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    cat_cols = categorical_columns(X)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("time", CyclicalTime(), [config.TIME_COL]),
            ("num", Pipeline(num_steps), config.NUMERIC_COLS),
            (
                "cat",
                Pipeline([
                    # Missing is informative here (e.g. 35% of 'Defect_of_vehicle'),
                    # so it becomes its own category instead of being filled with the mode.
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                             sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
