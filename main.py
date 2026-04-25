"""Fitaro ML pipeline orchestrator.

Run this script to train the model end-to-end:
    python main.py

Steps:
    1. Load and validate dataset.
    2. Preprocess features.
    3. Stratified train/test split.
    4. Train XGBoost with CV hyperparameter tuning.
    5. Evaluate on test set; save plots and reports.
    6. Generate SHAP global explanations; save plots.
    7. Run 3 sample predictions with text justifications.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on the path when running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    SIZE_ORDER,
    TEST_SIZE,
)
from src.data_loader import DataLoadError, SchemaValidationError, load_data
from src.evaluator import evaluate
from src.explainer import (
    build_explainer,
    plot_global_bar,
    plot_global_summary,
    plot_waterfall,
)
from src.model import predict_proba, save_model, train_model
from src.preprocessor import (
    fit_transform_train,
    save_preprocessor,
    save_preprocessor_versioned,
    transform_input,
    get_feature_names,
)
from src.predictor import FitaroPredictor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fitaro.main")


# ---------------------------------------------------------------------------
# Sample inputs for the demo predictions at the end of the run
# ---------------------------------------------------------------------------

SAMPLE_INPUTS = [
    {
        "Height_m": 1.75,
        "Weight_kg": 82,
        "Age": 30,
        "Chest_in": 42,
        "Length_in": 29,
        "Sleeve_in": 25,
        "ShoulderWidth_in": 18.5,
        "FitPreference": "Regular",
    },
    {
        "Height_m": 1.60,
        "Weight_kg": 55,
        "Age": 22,
        "Chest_in": 35,
        "Length_in": 26,
        "Sleeve_in": 22,
        "ShoulderWidth_in": 15.5,
        "FitPreference": "Slimfit",
    },
    {
        "Height_m": 1.85,
        "Weight_kg": 100,
        "Age": 45,
        "Chest_in": 48,
        "Length_in": 31,
        "Sleeve_in": 27,
        "ShoulderWidth_in": 20.5,
        "FitPreference": "Oversize",
    },
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full Fitaro training and evaluation pipeline."""
    trained_at = datetime.now()

    # Ensure output directories exist before anything writes to them.
    for d in (MODELS_DIR, PLOTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Loading data ===")
    try:
        df = load_data()
    except (DataLoadError, SchemaValidationError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Stratified train / test split
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Splitting data (%.0f%% test) ===", TEST_SIZE * 100)

    from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

    X_raw = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_raw = df[TARGET_COLUMN].values

    # Use SIZE_ORDER as the canonical mapping so SIZE_ORDER[i] always equals
    # the class whose XGBoost predict_proba column is at index i.
    size_to_int = {s: i for i, s in enumerate(SIZE_ORDER)}
    y = np.array([size_to_int[s] for s in y_raw])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(
        "Train: %d samples | Test: %d samples", len(X_train_raw), len(X_test_raw)
    )

    # ------------------------------------------------------------------
    # 3. Preprocess
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Preprocessing features ===")
    try:
        X_train, preprocessor = fit_transform_train(X_train_raw)
        X_test = transform_input(X_test_raw, preprocessor)
        feature_names = get_feature_names(preprocessor)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Train XGBoost with CV + hyperparameter search
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Training model ===")
    try:
        model, best_params = train_model(X_train, y_train)
    except Exception as exc:
        logger.error("Model training failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("=== Step 5: Evaluating on test set ===")
    try:
        y_pred = model.predict(X_test)
        results = evaluate(
            y_pred=y_pred,
            y_true=y_test,
            label_classes=SIZE_ORDER,
            trained_at=trained_at,
            best_params=best_params,
            save=True,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Save model and preprocessor
    # ------------------------------------------------------------------
    logger.info("=== Step 6: Saving model artefacts ===")
    save_model(model, trained_at=trained_at, best_params=best_params)
    save_preprocessor(preprocessor)
    save_preprocessor_versioned(preprocessor, trained_at=trained_at)

    # ------------------------------------------------------------------
    # 7. SHAP global explanations
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Generating SHAP global explanations ===")
    try:
        explainer = build_explainer(model)
        # Use a subsample of the test set to keep plot generation fast.
        X_explain = X_test[:200]
        plot_global_summary(explainer, X_explain, feature_names)
        plot_global_bar(explainer, X_explain, feature_names)
    except Exception as exc:
        logger.warning("SHAP global plots failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # 8. Sample predictions with text justifications
    # ------------------------------------------------------------------
    logger.info("=== Step 8: Sample predictions ===")
    try:
        predictor = FitaroPredictor(model=model, preprocessor=preprocessor)
        for i, sample in enumerate(SAMPLE_INPUTS, start=1):
            result = predictor.predict(sample)
            print(f"\n{'='*60}")
            print(f"Sample {i} | Input: {sample}")
            print(f"{'='*60}")
            print(result["justification"])
    except Exception as exc:
        logger.error("Sample predictions failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FITARO PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Adjacent-size error rate : {results['adjacent_error_rate'] * 100:.1f}%")
    print(f"  Non-adjacent error rate  : {results['non_adjacent_error_rate'] * 100:.1f}%")
    print(f"  Model saved to           : {MODELS_DIR}")
    print(f"  Plots saved to           : {PLOTS_DIR}")
    print(f"  Reports saved to         : {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
