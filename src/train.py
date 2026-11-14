"""Train, tune, compare and export the accident-severity model.

Usage:
    python -m src.train                    # full run (uses data/Addis_Ababa_city_RTA.csv)
    python -m src.train --data path.csv    # custom CSV
    python -m src.train --fast             # tiny search, for smoke tests / CI

Design choices (see README):
  * Preprocessing is part of the sklearn Pipeline -> no leakage inside CV.
  * Imbalance is handled with class/sample weights, NOT duplicate-oversampling
    (oversampling before CV puts copies of the same row in train and validation
    folds and produces over-optimistic, overfitted hyper-parameters).
  * Models are SELECTED by cross-validated macro-F1 on the training set; the test
    set is evaluated once at the end and never used for selection.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .data import clean, encode_target, label_mapping, load_data, train_test
from .evaluate import (
    compute_metrics, plot_confusion, plot_feature_importance, plot_model_comparison,
)
from .features import build_preprocessor, categorical_columns
from .postprocess import apply_weights, tune_class_weights


def build_candidates(X_train: pd.DataFrame, fast: bool, n_classes: int):
    rs = config.RANDOM_STATE
    tree_pre = lambda: build_preprocessor(X_train, scale_numeric=False)
    lin_pre = lambda: build_preprocessor(X_train, scale_numeric=True)

    candidates = {
        "Logistic Regression": (
            Pipeline([("prep", lin_pre()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
            {"clf__C": loguniform(1e-3, 1e1)},
            False,
        ),
        "Random Forest": (
            Pipeline([("prep", tree_pre()),
                      ("clf", RandomForestClassifier(random_state=rs, n_jobs=-1))]),
            {
                "clf__n_estimators": [100] if fast else [200, 400],
                "clf__max_depth": [None, 8, 12, 20],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__max_features": ["sqrt", "log2", 0.3],
                "clf__class_weight": ["balanced", "balanced_subsample"],
            },
            False,
        ),
        "XGBoost": (
            Pipeline([("prep", tree_pre()),
                      ("clf", XGBClassifier(objective="multi:softprob", num_class=n_classes,
                                            eval_metric="mlogloss", tree_method="hist",
                                            random_state=rs, n_jobs=1))]),
            {
                "clf__n_estimators": [50] if fast else [200, 400],
                "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.5, 0.7, 0.9],
                "clf__min_child_weight": [1, 3, 5],
                "clf__reg_lambda": [1, 5, 10],
            },
            True,  # needs sample_weight for class balancing
        ),
    }
    return candidates


def feature_names(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def main(data_path=None, fast: bool = False, n_iter: int | None = None) -> dict:
    t0 = time.time()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_data(data_path))
    mapping = label_mapping(df[config.TARGET])
    labels = list(mapping.values())
    print(f"Rows: {len(df)} | classes: {df[config.TARGET].value_counts().to_dict()}")

    X_train, X_test, y_train_raw, y_test_raw = train_test(df)
    y_train = encode_target(y_train_raw, mapping)
    y_test = encode_target(y_test_raw, mapping)

    cv = StratifiedKFold(n_splits=3 if fast else config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    n_iter = n_iter or (3 if fast else 25)
    sw_train = compute_sample_weight("balanced", y_train)

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    needs_sw_map: dict[str, bool] = {}

    # --- Baseline: always predict the majority class -------------------------------
    dummy = Pipeline([("prep", build_preprocessor(X_train)),
                      ("clf", DummyClassifier(strategy="most_frequent"))]).fit(X_train, y_train)
    results["Majority baseline"] = {
        "cv_macro_f1": None,
        "best_params": {},
        "test": compute_metrics(y_test, dummy.predict(X_test), labels),
    }
    fitted["Majority baseline"] = dummy

    # --- Candidates: tuned with CV on the TRAIN set only ---------------------------
    for name, (pipe, space, needs_sw) in build_candidates(X_train, fast, len(labels)).items():
        print(f"\n>>> Tuning {name} ...")
        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, scoring="f1_macro", cv=cv,
            n_jobs=-1, random_state=config.RANDOM_STATE, refit=True, error_score="raise",
        )
        fit_params = {"clf__sample_weight": sw_train} if needs_sw else {}
        search.fit(X_train, y_train, **fit_params)
        pred = search.predict(X_test)
        results[name] = {
            "cv_macro_f1": float(search.best_score_),
            "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()},
            "test": compute_metrics(y_test, pred, labels),
        }
        fitted[name] = search.best_estimator_
        needs_sw_map[name] = needs_sw
        print(f"    CV macro-F1 = {search.best_score_:.4f} | "
              f"test macro-F1 = {results[name]['test']['macro_f1']:.4f}")

    # --- Model selection: by CV score (never by test score) ------------------------
    contenders = {k: v for k, v in results.items() if v["cv_macro_f1"] is not None}
    best_name = max(contenders, key=lambda k: contenders[k]["cv_macro_f1"])
    best_pipe = fitted[best_name]
    print(f"\nSelected model: {best_name}")

    # --- Decision-weight tuning on out-of-fold TRAIN probabilities ------------------
    print("Tuning decision weights on out-of-fold predictions ...")
    oof = cross_val_predict(
        clone(best_pipe), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1,
        params={"clf__sample_weight": sw_train} if needs_sw_map[best_name] else None,
    )
    weights, oof_before, oof_after = tune_class_weights(y_train, oof)
    use_weights = (oof_after - oof_before) >= 0.005     # only keep it if the gain is real
    if not use_weights:
        weights = np.ones(len(labels))
    proba_test = best_pipe.predict_proba(X_test)
    best_pred = apply_weights(proba_test, weights)
    print(f"    OOF macro-F1: default {oof_before:.4f} -> tuned {oof_after:.4f} | "
          f"weights {np.round(weights, 3).tolist()} | applied: {use_weights}")
    if use_weights:
        results[f"{best_name} + tuned decision"] = {
            "cv_macro_f1": oof_after,
            "best_params": {"decision_weights": np.round(weights, 4).tolist()},
            "test": compute_metrics(y_test, best_pred, labels),
        }
    final_name = f"{best_name} + tuned decision" if use_weights else best_name

    # --- Reports -------------------------------------------------------------------
    plot_model_comparison(results, config.FIGURES_DIR / "model_comparison.png")
    plot_confusion(y_test, best_pred, labels, f"{final_name} - test set",
                   config.FIGURES_DIR / "confusion_matrix.png")
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        plot_feature_importance(feature_names(best_pipe), clf.feature_importances_,
                                config.FIGURES_DIR / "feature_importance.png")
    elif hasattr(clf, "coef_"):
        plot_feature_importance(feature_names(best_pipe), np.abs(clf.coef_).mean(axis=0),
                                config.FIGURES_DIR / "feature_importance.png")

    summary = {
        "selected_model": best_name,
        "final_model": final_name,
        "decision_weights": np.round(weights, 4).tolist(),
        "selection_criterion": "cross-validated macro-F1 on the training set",
        "results": results,
    }
    config.METRICS_PATH.write_text(json.dumps(summary, indent=2))

    # --- Export artifacts for deployment -------------------------------------------
    cat_cols = categorical_columns(X_train)
    metadata = {
        "model_name": final_name,
        "decision_weights": np.round(weights, 4).tolist(),
        "labels": {str(k): v for k, v in mapping.items()},
        "input_columns": list(X_train.columns),
        "categorical_options": {
            c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cat_cols
        },
        "numeric_ranges": {
            c: [int(df[c].min()), int(df[c].max())] for c in config.NUMERIC_COLS
        },
        "test_metrics": results[final_name]["test"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    joblib.dump(best_pipe, config.MODEL_PATH, compress=3)
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\n=== Test-set summary ===")
    table = pd.DataFrame({
        k: {"CV macro-F1": v["cv_macro_f1"], **{m: v["test"][m] for m in
            ("macro_f1", "balanced_accuracy", "accuracy", "mcc")}}
        for k, v in results.items()
    }).T.round(3)
    print(table.to_string())
    print(f"\nSaved: {config.MODEL_PATH.name}, {config.METADATA_PATH.name} "
          f"({time.time() - t0:.0f}s total)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="Path to the RTA csv")
    ap.add_argument("--fast", action="store_true", help="Tiny search for smoke tests")
    ap.add_argument("--n-iter", type=int, default=None, help="Random-search iterations per model")
    args = ap.parse_args()
    main(args.data, args.fast, args.n_iter)
