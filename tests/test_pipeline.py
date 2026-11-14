import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean, encode_target, label_mapping, train_test, validate
from src.features import CyclicalTime, build_preprocessor
from tests.synthetic import make_synthetic


@pytest.fixture(scope="module")
def df():
    return clean(make_synthetic(600, seed=1))


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate(pd.DataFrame({"a": [1]}))


def test_clean_drops_duplicates():
    d = make_synthetic(50)
    out = clean(pd.concat([d, d.iloc[:10]]))
    assert len(out) == len(d)


def test_split_is_stratified_and_disjoint(df):
    X_tr, X_te, y_tr, y_te = train_test(df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    ratio_tr = y_tr.value_counts(normalize=True)
    ratio_te = y_te.value_counts(normalize=True)
    assert np.allclose(ratio_tr.sort_index(), ratio_te.sort_index(), atol=0.03)


def test_label_mapping_is_ordered_by_severity(df):
    m = label_mapping(df[config.TARGET])
    assert list(m.values()) == config.CLASS_ORDER
    assert set(encode_target(df[config.TARGET], m)) == {0, 1, 2}


def test_cyclical_time_handles_bad_values():
    t = CyclicalTime().fit(pd.DataFrame({"Time": ["17:02:00", "1:06:00", "bad", None]}))
    out = t.transform(pd.DataFrame({"Time": ["0:00:00", "12:00:00", "garbage"]}))
    assert out.shape == (3, 3)
    assert np.isfinite(out).all()
    assert np.allclose(out[0, :2], [0, 1])          # midnight -> sin 0, cos 1
    assert out[0, 2] == 1 and out[1, 2] == 0          # night flag


def test_preprocessor_unseen_category_and_nan(df):
    X, _ = df.drop(columns=[config.TARGET]), df[config.TARGET]
    pre = build_preprocessor(X).fit(X)
    row = X.iloc[[0]].copy()
    row["Weather_conditions"] = "Hurricane"        # never seen in training
    row["Educational_level"] = np.nan
    out = pre.transform(row)
    assert out.shape[0] == 1 and np.isfinite(out).all()


def test_end_to_end_train_and_predict(tmp_path, monkeypatch):
    from src import predict, train

    csv = tmp_path / "rta.csv"
    make_synthetic(500, seed=3).to_csv(csv, index=False)
    for name in ("MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models_dir" / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "models_dir" / "metadata.json")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "reports_dir" / "metrics.json")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = train.main(csv, fast=True, n_iter=2)
    assert summary["selected_model"] in summary["results"]
    assert "Majority baseline" in summary["results"]

    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text())
    record = make_synthetic(5, seed=9).drop(columns=[config.TARGET]).iloc[0].to_dict()
    out = predict.predict_one(record, model, meta)
    assert out["prediction"] in config.CLASS_ORDER
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-6


def test_tune_class_weights_never_worse_than_argmax():
    from src.postprocess import apply_weights, tune_class_weights

    rng = np.random.default_rng(0)
    y = rng.choice(3, 600, p=[0.8, 0.15, 0.05])
    logits = rng.normal(size=(600, 3)) + np.eye(3)[y] * 1.2
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    w, before, after = tune_class_weights(y, proba)
    assert w[0] == 1.0 and len(w) == 3
    assert after >= before
    assert apply_weights(proba, np.ones(3)).shape == (600,)


def test_predict_uses_decision_weights():
    from src.predict import predict_one

    class Fake:
        def predict_proba(self, X):
            return np.array([[0.5, 0.3, 0.2]])

    meta = {"input_columns": ["a"], "labels": {"0": "Slight Injury", "1": "Serious Injury", "2": "Fatal injury"}}
    assert predict_one({"a": 1}, Fake(), {**meta, "decision_weights": [1, 1, 1]})["prediction"] == "Slight Injury"
    assert predict_one({"a": 1}, Fake(), {**meta, "decision_weights": [1, 2, 1]})["prediction"] == "Serious Injury"
