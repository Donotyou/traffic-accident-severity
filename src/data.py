"""Data loading, validation and train/test splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Find the CSV: explicit path -> data/ folder -> current working directory."""
    candidates = [Path(path)] if path else []
    candidates += [config.DATA_PATH, Path.cwd() / config.DATA_PATH.name]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Dataset not found. Place '{config.DATA_PATH.name}' in the data/ folder.\nTried:\n  {tried}"
    )


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and run basic sanity checks."""
    df = pd.read_csv(resolve_data_path(path))
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    required = [config.TARGET, config.TIME_COL, *config.NUMERIC_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[config.TARGET].isna().any():
        raise ValueError("Target column contains missing values.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and strip whitespace from text columns."""
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return X, y


def train_test(df: pd.DataFrame):
    """Stratified hold-out split. The test set is touched ONCE, at the very end."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def label_mapping(y: pd.Series) -> dict[int, str]:
    """Stable id -> label mapping ordered by severity (falls back to sorted labels)."""
    labels = set(y.unique())
    order = config.CLASS_ORDER if labels <= set(config.CLASS_ORDER) else sorted(labels)
    order = [c for c in order if c in labels]
    return dict(enumerate(order))


def encode_target(y: pd.Series, mapping: dict[int, str]) -> pd.Series:
    inverse = {v: k for k, v in mapping.items()}
    return y.map(inverse).astype(int)
