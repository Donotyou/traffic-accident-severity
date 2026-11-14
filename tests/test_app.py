"""Smoke test of the Streamlit UI using Streamlit's headless AppTest."""
import json
from pathlib import Path

import joblib
import pytest

from src import config
from tests.synthetic import make_synthetic

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_and_predicts(tmp_path, monkeypatch):
    from src import train
    csv = tmp_path / "rta.csv"
    make_synthetic(400, seed=5).to_csv(csv, index=False)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "fig")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    train.main(csv, fast=True, n_iter=2)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Predicted" in s.value for s in at.subheader)
