"""Model evaluation utilities for Fitaro.

Generates the classification report, confusion matrix heatmap, and the
project's key custom metric: adjacent-size error rate.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import ADJACENT_PAIRS, PLOTS_DIR, REPORTS_DIR, SIZE_ORDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_classes: List[str],
    trained_at: Optional[datetime] = None,
    best_params: Optional[Dict[str, Any]] = None,
    save: bool = True,
) -> Dict[str, object]:
    """Run the full evaluation suite and optionally persist results.

    Args:
        y_true: Integer ground-truth labels.
        y_pred: Integer predicted labels.
        label_classes: Class name strings in label-integer order (from LabelEncoder).
        save: Whether to write reports/plots to disk.

    Returns:
        Dictionary with keys: 'report_str', 'accuracy', 'macro_f1',
        'adjacent_error_rate', 'non_adjacent_error_rate'.
    """
    # Map integers back to size strings for human-readable output.
    true_str = [label_classes[i] for i in y_true]
    pred_str = [label_classes[i] for i in y_pred]

    report_dict = classification_report(
        true_str, pred_str, labels=SIZE_ORDER, zero_division=0, output_dict=True
    )
    report_str = classification_report(true_str, pred_str, labels=SIZE_ORDER, zero_division=0)
    logger.info("\nClassification Report:\n%s", report_str)

    adj_rate, non_adj_rate = _adjacent_size_error_rate(true_str, pred_str)
    acc = float(accuracy_score(true_str, pred_str))
    macro_f1 = float(f1_score(true_str, pred_str, average="macro", labels=SIZE_ORDER, zero_division=0))

    results = {
        "report_str": report_str,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "adjacent_error_rate": adj_rate,
        "non_adjacent_error_rate": non_adj_rate,
    }

    logger.info(
        "\n=== Adjacent-Size Error Rate: %.1f%% ===\n"
        "    Non-Adjacent Error Rate:   %.1f%%\n"
        "    (Among all misclassifications)",
        adj_rate * 100,
        non_adj_rate * 100,
    )

    if save:
        _save_confusion_matrix(true_str, pred_str)
        _save_confusion_matrix_normalized(true_str, pred_str)
        _save_per_class_metrics(report_dict)
        _save_report(report_str, adj_rate, non_adj_rate, acc, macro_f1, trained_at, best_params)

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _adjacent_size_error_rate(
    true_str: List[str], pred_str: List[str]
) -> tuple[float, float]:
    """Calculate what fraction of mistakes are between neighbouring sizes.

    Adjacent errors (S↔M, M↔L, …) are much less harmful than distant errors
    (S↔XXL), so tracking this separately gives a more nuanced quality signal.

    Args:
        true_str: Ground-truth size labels as strings.
        pred_str: Predicted size labels as strings.

    Returns:
        Tuple of (adjacent_rate, non_adjacent_rate) where both are fractions
        of total misclassifications. Rates sum to 1.0.
    """
    errors = [(t, p) for t, p in zip(true_str, pred_str) if t != p]
    total_errors = len(errors)

    if total_errors == 0:
        logger.info("Perfect predictions — no misclassifications to analyse.")
        return 0.0, 0.0

    adjacent_errors = sum(1 for t, p in errors if (t, p) in ADJACENT_PAIRS)
    adj_rate = adjacent_errors / total_errors
    non_adj_rate = 1.0 - adj_rate

    logger.debug(
        "%d total misclassifications: %d adjacent (%.1f%%), %d non-adjacent (%.1f%%)",
        total_errors, adjacent_errors, adj_rate * 100,
        total_errors - adjacent_errors, non_adj_rate * 100,
    )
    return adj_rate, non_adj_rate


def _save_confusion_matrix(true_str: List[str], pred_str: List[str]) -> None:
    """Generate and save the confusion matrix as a PNG heatmap."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(true_str, pred_str, labels=SIZE_ORDER)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SIZE_ORDER,
        yticklabels=SIZE_ORDER,
        ax=ax,
    )
    ax.set_xlabel("Predicted Size", fontsize=12)
    ax.set_ylabel("True Size", fontsize=12)
    ax.set_title("Confusion Matrix — Fitaro Size Prediction", fontsize=14)

    out_path = PLOTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)


def _save_confusion_matrix_normalized(true_str: List[str], pred_str: List[str]) -> None:
    """Generate and save the row-normalized confusion matrix."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(true_str, pred_str, labels=SIZE_ORDER)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=SIZE_ORDER,
        yticklabels=SIZE_ORDER,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Size", fontsize=12)
    ax.set_ylabel("True Size", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized) — Fitaro Size Prediction", fontsize=14)

    out_path = PLOTS_DIR / "confusion_matrix_normalized.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Normalized confusion matrix saved to %s", out_path)


def _save_per_class_metrics(report_dict: Dict[str, Any]) -> None:
    """Save per-class precision/recall/F1 chart."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in SIZE_ORDER:
        if cls not in report_dict:
            continue
        rows.append(
            {
                "class": cls,
                "precision": float(report_dict[cls].get("precision", 0.0)),
                "recall": float(report_dict[cls].get("recall", 0.0)),
                "f1": float(report_dict[cls].get("f1-score", 0.0)),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df["class"]))
    w = 0.25
    ax.bar(x - w, df["precision"], width=w, label="Precision")
    ax.bar(x, df["recall"], width=w, label="Recall")
    ax.bar(x + w, df["f1"], width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class metrics — Fitaro")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    out_path = PLOTS_DIR / "per_class_metrics.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Per-class metrics plot saved to %s", out_path)


def _save_report(
    report_str: str,
    adj_rate: float,
    non_adj_rate: float,
    accuracy: float,
    macro_f1: float,
    trained_at: Optional[datetime],
    best_params: Optional[Dict[str, Any]],
) -> None:
    """Write the classification report and adjacent-error summary to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trained_line = ""
    if trained_at:
        trained_line = f"Trained at               : {trained_at.isoformat(timespec='seconds')}\n"

    best_params_block = ""
    if best_params:
        best_params_block = "Best hyperparameters:\n" + "\n".join(
            [f"  - {k}: {v}" for k, v in best_params.items()]
        ) + "\n\n"

    summary = (
        "=== Fitaro Model Evaluation Report ===\n\n"
        + trained_line
        + f"Accuracy                 : {accuracy * 100:.2f}%\n"
        + f"Macro-F1                 : {macro_f1:.4f}\n\n"
        + best_params_block
        + f"{report_str}\n"
        "=== Adjacent-Size Error Analysis ===\n"
        f"Adjacent-size error rate  : {adj_rate * 100:.1f}%\n"
        f"Non-adjacent error rate   : {non_adj_rate * 100:.1f}%\n"
        "(Fractions are of total misclassifications)\n"
    )

    out_path = REPORTS_DIR / "evaluation_report.txt"
    out_path.write_text(summary, encoding="utf-8")
    logger.info("Evaluation report saved to %s", out_path)
