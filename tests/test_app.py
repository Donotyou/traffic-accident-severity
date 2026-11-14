"""Smoke test of the Streamlit UI using Streamlit's headless AppTest."""
from pathlib import Path

import pytest

from src import config
from tests.synthetic import make_synthetic

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture()
def trained(tmp_path, monkeypatch):
    from src import train

    csv = tmp_path / "rta.csv"
    make_synthetic(400, seed=5).to_csv(csv, index=False)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "fig")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    train.main(csv, fast=True, n_iter=2)


def test_app_renders_and_predicts_live(trained):
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    html_blocks = " ".join(m.value for m in at.markdown)
    assert "Predicted severity" in html_blocks          # live prediction is rendered
    assert len(at.selectbox) >= 20


def test_random_example_and_reset_buttons(trained):
    at = AppTest.from_file(APP, default_timeout=90).run()
    labels = [b.label for b in at.sidebar.button]
    assert "Load random example" in labels and "Reset all inputs" in labels
    at.sidebar.button[labels.index("Load random example")].click().run()
    assert not at.exception
    at.sidebar.button[labels.index("Reset all inputs")].click().run()
    assert not at.exception
