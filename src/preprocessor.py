"""Feature preprocessing pipeline for Fitaro.

Builds a sklearn ColumnTransformer that:
- Scales numeric features with StandardScaler.
- One-hot encodes the FitPreference categorical feature.

The fitted preprocessor is saved to disk so predict.py can reuse it without
re-fitting on unseen data.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    LATEST_MODEL_META_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_SAVE_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Construct (but do not fit) the ColumnTransformer pipeline.

    Returns:
        An unfitted ColumnTransformer ready for fit_transform / transform.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Silently discard any extra columns.
    )
    return preprocessor


def fit_transform_train(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, ColumnTransformer]:
    """Fit the preprocessor on training data and return transformed features.

    Target encoding is intentionally left to main.py so the caller controls
    the class→integer mapping and avoids ordering ambiguity.

    Args:
        df: Training DataFrame (target column is ignored here).

    Returns:
        Tuple of (X_transformed, fitted_preprocessor).
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    logger.info("Preprocessor fitted. Output shape: %s.", X_transformed.shape)
    return X_transformed, preprocessor


def transform_input(
    raw_input: pd.DataFrame, preprocessor: ColumnTransformer
) -> np.ndarray:
    """Transform a new (unseen) sample using the already-fitted preprocessor.

    Args:
        raw_input: DataFrame with the same feature columns as training data.
        preprocessor: The fitted ColumnTransformer from fit_transform_train.

    Returns:
        Transformed feature array ready for model inference.
    """
    return preprocessor.transform(raw_input)


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = PREPROCESSOR_SAVE_PATH) -> None:
    """Persist the fitted preprocessor to disk.

    Args:
        preprocessor: Fitted ColumnTransformer to save.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor saved to %s", path)


def _read_latest_meta() -> Optional[Dict[str, Any]]:
    try:
        if not LATEST_MODEL_META_PATH.exists():
            return None
        return json.loads(LATEST_MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_preprocessor_versioned(
    preprocessor: ColumnTransformer,
    trained_at: datetime,
    path: Path = PREPROCESSOR_SAVE_PATH,
) -> Path:
    """Save a timestamped copy of the preprocessor and update latest metadata."""
    slug = trained_at.strftime("%Y%m%d_%H%M%S")
    ts_path = path.with_name(f"{path.stem}_{slug}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ts_path)

    meta = _read_latest_meta() or {}
    meta.update(
        {
            "preprocessor_latest_path": str(path),
            "preprocessor_timestamped_path": str(ts_path),
        }
    )
    LATEST_MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Preprocessor (timestamped) saved to %s", ts_path)
    return ts_path


def load_preprocessor(path: Path = PREPROCESSOR_SAVE_PATH) -> ColumnTransformer:
    """Load a previously saved preprocessor from disk.

    Args:
        path: Path to the saved .joblib file.

    Returns:
        The fitted ColumnTransformer.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    meta = _read_latest_meta()
    if meta and meta.get("preprocessor_timestamped_path"):
        meta_path = Path(meta["preprocessor_timestamped_path"])
        if meta_path.exists():
            preprocessor = joblib.load(meta_path)
            logger.info("Preprocessor loaded from %s", meta_path)
            return preprocessor

    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run main.py first.")
    preprocessor = joblib.load(path)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the feature names produced by the fitted ColumnTransformer.

    This is used by the explainer to map SHAP values back to human-readable
    feature names.

    Args:
        preprocessor: A fitted ColumnTransformer.

    Returns:
        List of feature name strings in the same order as transformed columns.
    """
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_names
