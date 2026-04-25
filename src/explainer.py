"""SHAP-based explainability for Fitaro size recommendations.

Provides:
- Global feature importance plots (beeswarm + bar).
- Local per-prediction plots (waterfall + force).
- Human-readable text justification that non-technical users can understand.

Note on SHAP output format:
    Newer SHAP (>=0.42) + XGBoost returns a 3D ndarray of shape
    (n_samples, n_features, n_classes).  Older versions returned a list of
    2D arrays, one per class.  The private _shap_per_class() helper
    normalises both to the list-of-2D form used throughout this module.
"""

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving.
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBClassifier

from src.config import PLOTS_DIR, SIZE_ORDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP normalisation helper
# ---------------------------------------------------------------------------


def _shap_per_class(raw: object) -> List[np.ndarray]:
    """Normalise SHAP output to a list of 2D arrays, one per class.

    Args:
        raw: Output of ``TreeExplainer.shap_values()`` — either a list of 2D
            arrays or a single 3D array (n_samples, n_features, n_classes).

    Returns:
        List where item [i] has shape (n_samples, n_features).
    """
    if isinstance(raw, list):
        return raw  # Already in the expected format.
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # Newer SHAP: (n_samples, n_features, n_classes) → list of 2D slices.
        n_classes = raw.shape[2]
        return [raw[:, :, i] for i in range(n_classes)]
    # Fallback: assume 2D (binary / single output).
    return [raw]


# ---------------------------------------------------------------------------
# SHAP explainer initialisation
# ---------------------------------------------------------------------------


def build_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the trained XGBoost model.

    TreeExplainer is exact and fast for tree-based models — no approximations
    needed unlike KernelExplainer.

    Args:
        model: A fitted XGBClassifier.

    Returns:
        Initialised (but not yet called) TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer initialised.")
    return explainer


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------


def plot_global_summary(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: List[str],
) -> None:
    """Save a SHAP beeswarm summary plot for the dataset.

    Averages absolute SHAP values across all classes to give a model-level
    view of which features drive predictions.

    Args:
        explainer: Fitted TreeExplainer.
        X_transformed: Preprocessed feature matrix for the test set.
        feature_names: Human-readable names in column order.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = explainer.shap_values(X_transformed)
    shap_list = _shap_per_class(raw)

    # Average |SHAP| across classes → shape (n_samples, n_features).
    shap_agg = np.mean(np.abs(np.array(shap_list)), axis=0)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_agg,
        X_transformed,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Fitaro (all sizes)", pad=10)
    out_path = PLOTS_DIR / "shap_summary_beeswarm.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP beeswarm plot saved to %s", out_path)


def plot_global_bar(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: List[str],
) -> None:
    """Save a SHAP bar plot of mean absolute feature importance.

    Args:
        explainer: Fitted TreeExplainer.
        X_transformed: Preprocessed feature matrix.
        feature_names: Human-readable names in column order.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = explainer.shap_values(X_transformed)
    shap_list = _shap_per_class(raw)

    # Mean |SHAP| per feature, averaged across samples AND classes.
    mean_abs_per_class = [np.mean(np.abs(sv), axis=0) for sv in shap_list]
    mean_abs = np.mean(np.array(mean_abs_per_class), axis=0)  # shape (n_features,)

    sorted_idx = np.argsort(mean_abs)  # ascending — barh plots bottom-to-top

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx.tolist()],
        mean_abs[sorted_idx],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (Mean |SHAP|) — Fitaro")
    fig.tight_layout()

    out_path = PLOTS_DIR / "shap_bar_global.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Local (per-prediction) explanations
# ---------------------------------------------------------------------------


def plot_waterfall(
    explainer: shap.TreeExplainer,
    X_single: np.ndarray,
    feature_names: List[str],
    predicted_class_idx: int,
    sample_label: str = "sample",
) -> None:
    """Save a SHAP waterfall plot for one prediction.

    Args:
        explainer: Fitted TreeExplainer.
        X_single: Single-row feature matrix (shape 1 × n_features).
        feature_names: Human-readable names in column order.
        predicted_class_idx: Index of the predicted class in SIZE_ORDER.
        sample_label: String used in the output filename.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = explainer.shap_values(X_single)
    shap_list = _shap_per_class(raw)

    sv = shap_list[predicted_class_idx][0]  # 1D, shape (n_features,)

    ev = explainer.expected_value
    base = ev[predicted_class_idx] if hasattr(ev, "__len__") else float(ev)

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_single[0],
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall — {sample_label} → {SIZE_ORDER[predicted_class_idx]}")
    out_path = PLOTS_DIR / f"shap_waterfall_{sample_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Waterfall plot saved to %s", out_path)


def plot_force(
    explainer: shap.TreeExplainer,
    X_single: np.ndarray,
    feature_names: List[str],
    predicted_class_idx: int,
    sample_label: str = "sample",
) -> None:
    """Save a SHAP force plot for one prediction as a PNG.

    Args:
        explainer: Fitted TreeExplainer.
        X_single: Single-row feature matrix.
        feature_names: Human-readable names in column order.
        predicted_class_idx: Index of the predicted class.
        sample_label: String used in the output filename.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = explainer.shap_values(X_single)
    shap_list = _shap_per_class(raw)

    sv = shap_list[predicted_class_idx][0]
    ev = explainer.expected_value
    base = ev[predicted_class_idx] if hasattr(ev, "__len__") else float(ev)

    shap.initjs()
    shap.force_plot(
        base,
        sv,
        X_single[0],
        feature_names=feature_names,
        show=False,
        matplotlib=True,
    )
    out_path = PLOTS_DIR / f"shap_force_{sample_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Force plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Text justification (star feature)
# ---------------------------------------------------------------------------


def generate_justification(
    shap_list: List[np.ndarray],
    predicted_class_idx: int,
    probabilities: np.ndarray,
    feature_names: List[str],
    raw_input: dict,
) -> str:
    """Convert SHAP values into a natural-language size recommendation.

    The goal is a concise, friendly explanation that a non-technical shopper
    can actually understand — not a data-science report.

    Args:
        shap_list: List of 2D SHAP arrays (one per class), each (1, n_features).
                   Use _shap_per_class() to obtain this from raw explainer output.
        predicted_class_idx: Index of the winning class in SIZE_ORDER.
        probabilities: Model output probabilities for all classes (1D array).
        feature_names: Names of features in the same order as SHAP columns.
        raw_input: Original user-supplied dict (for showing real measurement values).

    Returns:
        Multi-line justification string.
    """
    predicted_size = SIZE_ORDER[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    # Second-best class for the "next closest" line.
    sorted_class_idx = np.argsort(probabilities)[::-1]
    second_idx = int(sorted_class_idx[1])
    second_size = SIZE_ORDER[second_idx]
    second_conf = probabilities[second_idx]

    # SHAP values for the predicted class, first (only) sample.
    sv = shap_list[predicted_class_idx][0]  # shape (n_features,)

    # Pair feature names with SHAP values.
    feature_info = list(zip(feature_names, sv.tolist()))

    # For one-hot-encoded FitPreference columns, keep only the active column
    # (the one whose value is 1 for this user).  Showing inactive dummy columns
    # in the explanation confuses non-technical users.
    active_pref = raw_input.get("FitPreference", "")
    feature_info = [
        (name, val) for name, val in feature_info
        if not (name.startswith("FitPreference_") and name.split("_", 1)[1] != active_pref)
    ]

    # Sort by |SHAP| descending — most influential features first.
    feature_info.sort(key=lambda x: abs(x[1]), reverse=True)

    lines = [
        f"Recommended size: {predicted_size} (confidence: {confidence * 100:.0f}%)",
        "",
        "Why this size?",
    ]

    # Features with |SHAP| below this fraction of the max are "minimal".
    max_shap = max(abs(v) for _, v in feature_info) or 1.0
    threshold = 0.10 * max_shap

    influential = [(n, v) for n, v in feature_info if abs(v) >= threshold]
    minimal = [(n, v) for n, v in feature_info if abs(v) < threshold]

    for name, shap_val in influential:
        direction = "pushed toward" if shap_val > 0 else "pushed away from"
        label = _friendly_feature_name(name)
        val_str = _format_feature_value(name, raw_input)
        lines.append(f"  -> {label}{val_str} {direction} size {predicted_size}")

    if minimal:
        minimal_labels = ", ".join(_friendly_feature_name(n) for n, _ in minimal)
        lines.append(f"  -> {minimal_labels} had minimal influence")

    lines += [
        "",
        f"Next closest size: {second_size} ({second_conf * 100:.0f}%)",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_FEATURE_LABELS = {
    "Height_m": "Height",
    "Weight_kg": "Weight",
    "Age": "Age",
    "Chest_in": "Chest measurement",
    "Length_in": "Body length",
    "Sleeve_in": "Sleeve length",
    "ShoulderWidth_in": "Shoulder width",
    "FitPreference_Regular": "Regular fit preference",
    "FitPreference_Slimfit": "Slim-fit preference",
    "FitPreference_Oversize": "Oversize fit preference",
}


def _friendly_feature_name(name: str) -> str:
    """Convert internal feature names to readable labels."""
    return _FEATURE_LABELS.get(name, name)


def _format_feature_value(name: str, raw_input: dict) -> str:
    """Build a parenthesised value string, e.g. ' (42 in)'."""
    if name.startswith("FitPreference_"):
        # Only annotate the OHE column that matches the user's actual preference;
        # the others are already described by their label.
        pref = raw_input.get("FitPreference", "")
        col_pref = name.split("_", 1)[1]  # e.g. "Oversize" from "FitPreference_Oversize"
        return f" ({pref})" if col_pref == pref else ""

    val = raw_input.get(name)
    if val is None:
        return ""

    if name == "Height_m":
        return f" ({val:.2f} m)"
    if name == "Weight_kg":
        return f" ({val:.0f} kg)"
    if name == "Age":
        return f" ({int(val)} yrs)"
    if name.endswith("_in"):
        return f" ({val:.1f} in)"
    return f" ({val})"
