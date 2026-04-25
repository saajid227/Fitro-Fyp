"""End-to-end prediction pipeline for Fitaro.

Loads the saved model and preprocessor, validates raw user input, runs
inference, and returns the size recommendation with a full text justification.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import shap

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_BOUNDS,
    MODEL_SAVE_PATH,
    NUMERIC_FEATURES,
    PREPROCESSOR_SAVE_PATH,
    SIZE_ORDER,
    VALID_FIT_PREFERENCES,
)
from src.explainer import _shap_per_class, build_explainer, generate_justification
from src.model import load_model
from src.preprocessor import get_feature_names, load_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class InputValidationError(ValueError):
    """Raised when user-supplied measurement values are invalid."""


def validate_input(raw: Dict) -> None:
    """Check that all required fields are present and within sensible ranges.

    Args:
        raw: User-supplied dictionary of measurements.

    Raises:
        InputValidationError: On any missing field or out-of-range value.
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [f for f in required if f not in raw]
    if missing:
        raise InputValidationError(f"Missing required fields: {missing}")

    for col, (lo, hi) in FEATURE_BOUNDS.items():
        val = raw[col]
        if not (lo <= val <= hi):
            raise InputValidationError(
                f"'{col}' value {val} is outside the expected range [{lo}, {hi}]."
            )

    pref = raw.get("FitPreference")
    if pref not in VALID_FIT_PREFERENCES:
        raise InputValidationError(
            f"'FitPreference' must be one of {VALID_FIT_PREFERENCES}, got '{pref}'."
        )


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------


class FitaroPredictor:
    """Wraps the trained model and preprocessor for inference.

    Attributes:
        model: Fitted XGBClassifier.
        preprocessor: Fitted ColumnTransformer.
        explainer: SHAP TreeExplainer for the model.
        feature_names: Column names in transformed feature space.
    """

    def __init__(
        self,
        model: Optional[XGBClassifier] = None,
        preprocessor: Optional[ColumnTransformer] = None,
    ) -> None:
        """Initialise by loading saved artefacts if not supplied directly.

        Args:
            model: Pre-loaded XGBClassifier (optional).
            preprocessor: Pre-loaded ColumnTransformer (optional).
        """
        self.model = model or load_model(MODEL_SAVE_PATH)
        self.preprocessor = preprocessor or load_preprocessor(PREPROCESSOR_SAVE_PATH)
        self.explainer = build_explainer(self.model)
        self.feature_names = get_feature_names(self.preprocessor)
        logger.info("FitaroPredictor ready.")

    def predict(self, raw_input: Dict) -> Dict:
        """Produce a size recommendation with justification for a single user.

        Args:
            raw_input: Dictionary with keys matching NUMERIC_FEATURES +
                CATEGORICAL_FEATURES, e.g.
                {"Height_m": 1.75, "Weight_kg": 82, ..., "FitPreference": "Regular"}

        Returns:
            Dictionary with keys:
                - "size": Predicted size string (e.g. "L").
                - "confidence": Probability of the predicted size (float).
                - "probabilities": Dict mapping each size to its probability.
                - "justification": Natural-language explanation string.

        Raises:
            InputValidationError: If raw_input fails validation.
        """
        validate_input(raw_input)

        # Build a single-row DataFrame for the preprocessor.
        input_df = pd.DataFrame([raw_input])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_transformed = self.preprocessor.transform(input_df)

        # XGBoost returns probabilities in label-integer order; SIZE_ORDER
        # matches LabelEncoder's alphabetical order for these class names.
        proba = self.model.predict_proba(X_transformed)[0]
        predicted_idx = int(np.argmax(proba))
        predicted_size = SIZE_ORDER[predicted_idx]
        confidence = float(proba[predicted_idx])

        prob_dict = {size: float(proba[i]) for i, size in enumerate(SIZE_ORDER)}

        # SHAP local explanation.
        raw_shap = self.explainer.shap_values(X_transformed)
        shap_list = _shap_per_class(raw_shap)

        justification = generate_justification(
            shap_list=shap_list,
            predicted_class_idx=predicted_idx,
            probabilities=proba,
            feature_names=self.feature_names,
            raw_input=raw_input,
        )

        return {
            "size": predicted_size,
            "confidence": confidence,
            "probabilities": prob_dict,
            "justification": justification,
        }
