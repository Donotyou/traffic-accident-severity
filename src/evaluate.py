"""Evaluation metrics and plots.

Accuracy alone is misleading on this dataset (~83% of accidents are 'Slight Injury',
so a model that always predicts that class already scores ~83%). We therefore report
macro-F1, balanced accuracy, MCC and per-class recall, and always compare to a
majority-class baseline.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    ids = list(range(len(labels)))
    report = classification_report(
        y_true, y_pred, labels=ids, target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            lab: {k: float(report[lab][k]) for k in ("precision", "recall", "f1-score", "support")}
            for lab in labels
        },
    }


def plot_confusion(y_true, y_pred, labels, title, path):
    ids = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Counts")
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", xticklabels=labels, yticklabels=labels,
                ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Row-normalised (recall per class)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict], path):
    names = list(results)
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, [results[n]["test"][m] for n in names], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set comparison (accuracy is inflated by class imbalance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names, importances, path, top=20):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=np.asarray(importances)[order], y=np.asarray(names)[order], color="#3b6fb6", ax=ax)
    ax.set_title(f"Top {top} feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
