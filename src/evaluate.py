"""
Evaluation utilities:
  - compute_all_metrics  : accuracy, F1-macro, AUC-ROC
  - print_metrics        : formatted console output
  - plot_confusion_matrix: count + normalised heatmaps
  - plot_roc_curves      : per-class OvR ROC curves
  - plot_training_curves : loss and F1 vs epoch
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_all_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Args:
        labels      : (N,) ground-truth class indices
        preds       : (N,) predicted class indices
        probs       : (N, C) predicted class probabilities (softmax output)
        class_names : ordered list of class name strings

    Returns:
        dict with keys: accuracy, f1_macro, f1_per_class, auc_macro, auc_per_class
    """
    acc      = float(accuracy_score(labels, preds))
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_per   = f1_score(labels, preds, average=None, zero_division=0)

    # One-hot labels for AUC-ROC
    n_classes   = len(class_names)
    labels_ohe  = np.eye(n_classes)[labels]

    try:
        auc_macro = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        auc_per   = roc_auc_score(labels_ohe, probs, average=None)
        auc_per   = [float(v) for v in auc_per]
    except Exception as e:
        print(f"[WARNING] AUC-ROC computation failed: {e}")
        auc_macro = None
        auc_per   = [None] * n_classes

    return {
        "accuracy":      acc,
        "f1_macro":      f1_macro,
        "f1_per_class":  dict(zip(class_names, [float(v) for v in f1_per])),
        "auc_macro":     auc_macro,
        "auc_per_class": dict(zip(class_names, auc_per)),
        "report":        classification_report(labels, preds, target_names=class_names, zero_division=0),
    }


def print_metrics(metrics: Dict, prefix: str = "Test") -> None:
    w = 55
    print(f"\n{'═' * w}")
    print(f"  {prefix} Results")
    print(f"{'═' * w}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  F1-macro   : {metrics['f1_macro']:.4f}   (baseline: 0.738)")
    if metrics["auc_macro"] is not None:
        print(f"  AUC-macro  : {metrics['auc_macro']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<22} {'F1':>6}  {'AUC':>6}")
    print(f"  {'-'*38}")
    for cls in metrics["f1_per_class"]:
        f1  = metrics["f1_per_class"][cls]
        auc = metrics["auc_per_class"].get(cls)
        auc_str = f"{auc:.4f}" if auc is not None else "  N/A"
        print(f"  {cls:<22} {f1:>6.4f}  {auc_str:>6}")
    print(f"{'═' * w}")
    print(f"\n  Full classification report:\n")
    print(metrics["report"])


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Save side-by-side confusion matrices: raw counts and row-normalised.
    """
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Confusion Matrix — Counts"),
        (axes[1], cm_norm, ".2f",  "Confusion Matrix — Normalised"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title(title,        fontsize=13)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix saved → {output_path}")


def plot_roc_curves(
    labels:      np.ndarray,
    probs:       np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Save one-vs-rest ROC curves for each class.
    """
    n_classes  = len(class_names)
    labels_ohe = np.eye(n_classes)[labels]
    colors     = plt.cm.Set1(np.linspace(0.0, 0.85, n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_ohe[:, i], probs[:, i])
        try:
            auc = roc_auc_score(labels_ohe[:, i], probs[:, i])
            label = f"{cls}  (AUC = {auc:.3f})"
        except Exception:
            label = cls
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — One-vs-Rest",   fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] ROC curves saved → {output_path}")


def plot_training_curves(history: Dict, output_path: str) -> None:
    """
    Save training and validation loss + F1-macro curves over epochs.
    A dashed red line marks the 0.738 baseline for reference.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", markersize=3, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "s-", markersize=3, label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # F1-macro
    axes[1].plot(epochs, history["train_f1"], "o-", markersize=3, label="Train")
    axes[1].plot(epochs, history["val_f1"],   "s-", markersize=3, label="Val")
    axes[1].axhline(y=0.738, color="red", linestyle="--", linewidth=1.5, label="Baseline (0.738)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1-macro")
    axes[1].set_title("F1-macro over Epochs"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Training curves saved → {output_path}")
