"""Inference helpers shared by the Streamlit app and the CLI."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .postprocess import apply_weights


def _register_notebook_classes() -> None:
    """Let models exported from the notebook load anywhere.

    A pickle made inside a notebook points at classes defined in ``__main__``. We alias
    those names to the identical classes in ``src.features`` so the same model.joblib works
    in Streamlit, scripts and Docker, whichever way it was produced.
    """
    import __main__

    from .features import CyclicalTime

    if not hasattr(__main__, "CyclicalTime"):
        __main__.CyclicalTime = CyclicalTime


@lru_cache(maxsize=1)
def load_artifacts(model_path=None, metadata_path=None):
    _register_notebook_classes()
    model = joblib.load(model_path or config.MODEL_PATH)
    meta = json.loads((metadata_path or config.METADATA_PATH).read_text())
    return model, meta


def predict_one(record: dict, model=None, meta=None) -> dict:
    """Predict severity for one accident record (dict of column -> value)."""
    if model is None or meta is None:
        model, meta = load_artifacts()
    row = pd.DataFrame([record]).reindex(columns=meta["input_columns"])
    proba = model.predict_proba(row)[0]
    labels = [meta["labels"][str(i)] for i in range(len(proba))]
    weights = meta.get("decision_weights") or [1.0] * len(proba)
    best = int(apply_weights(proba.reshape(1, -1), weights)[0])
    return {
        "prediction": labels[best],
        "probabilities": dict(zip(labels, map(float, proba))),
    }
