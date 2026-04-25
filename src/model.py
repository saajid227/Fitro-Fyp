"""XGBoost model training, tuning, saving, and loading for Fitaro.

Uses RandomizedSearchCV with stratified 5-fold CV and macro-F1 scoring to
find good hyperparameters without exhaustive grid search.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    HYPERPARAM_GRID,
    LATEST_MODEL_META_PATH,
    MODEL_SAVE_PATH,
    N_ITER_SEARCH,
    RANDOM_SEED,
    SIZE_ORDER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train an XGBoost classifier with randomised hyperparameter search.

    Stratified k-fold ensures each fold preserves the class distribution,
    which matters here because XXL is the least-represented size.

    Args:
        X_train: Preprocessed training features.
        y_train: Integer-encoded size labels.

    Returns:
        Tuple of (best_estimator, best_params_dict).
    """
    n_classes = len(SIZE_ORDER)

    base_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=HYPERPARAM_GRID,
        n_iter=N_ITER_SEARCH,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    logger.info(
        "Starting RandomizedSearchCV with %d iterations over %d-fold CV.",
        N_ITER_SEARCH,
        CV_FOLDS,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_

    logger.info("Best CV macro-F1: %.4f", best_score)
    logger.info("Best hyperparameters: %s", best_params)

    return search.best_estimator_, best_params


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _safe_timestamp_slug(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")


def _write_latest_meta(meta: Dict[str, Any]) -> None:
    LATEST_MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_latest_meta() -> Optional[Dict[str, Any]]:
    try:
        if not LATEST_MODEL_META_PATH.exists():
            return None
        return json.loads(LATEST_MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_model(
    model: XGBClassifier,
    path: Path = MODEL_SAVE_PATH,
    trained_at: Optional[datetime] = None,
    best_params: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the trained model.

    Writes:
    - the "latest" stable path (`MODEL_SAVE_PATH`) for backward compatibility
    - a timestamp-suffixed copy for traceability
    - `latest_model.json` pointing to the newest artefacts

    Returns:
        The timestamped model path that was written.
    """
    trained_at = trained_at or datetime.now()
    slug = _safe_timestamp_slug(trained_at)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the old path working (predict.py / existing code may rely on it).
    joblib.dump(model, path)

    ts_path = path.with_name(f"{path.stem}_{slug}{path.suffix}")
    joblib.dump(model, ts_path)

    meta = _read_latest_meta() or {}
    meta.update(
        {
            "trained_at": trained_at.isoformat(timespec="seconds"),
            "model_latest_path": str(path),
            "model_timestamped_path": str(ts_path),
            "best_params": best_params or meta.get("best_params"),
        }
    )
    _write_latest_meta(meta)

    logger.info("Model saved to %s", path)
    logger.info("Model (timestamped) saved to %s", ts_path)
    return ts_path


def load_model(path: Path = MODEL_SAVE_PATH) -> XGBClassifier:
    """Load a previously saved model from disk.

    Args:
        path: Path to the saved .joblib file.

    Returns:
        The fitted XGBClassifier.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Prefer the newest timestamped artefact if available.
    meta = _read_latest_meta()
    if meta and meta.get("model_timestamped_path"):
        meta_path = Path(meta["model_timestamped_path"])
        if meta_path.exists():
            model = joblib.load(meta_path)
            logger.info("Model loaded from %s", meta_path)
            return model

    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run main.py first.")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return class probabilities for every sample in X.

    Args:
        model: A fitted XGBClassifier.
        X: Preprocessed feature matrix (n_samples, n_features).

    Returns:
        Array of shape (n_samples, n_classes) with probabilities summing to 1.
    """
    return model.predict_proba(X)
