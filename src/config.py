"""Central configuration for the Fitaro ML pipeline.

All constants, file paths, feature definitions, and hyperparameter grids live
here so other modules never contain magic strings or numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "garment_size_dataset_1000_asian_realistic.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

MODEL_SAVE_PATH = MODELS_DIR / "fitaro_xgb_model.joblib"
PREPROCESSOR_SAVE_PATH = MODELS_DIR / "fitaro_preprocessor.joblib"
LATEST_MODEL_META_PATH = MODELS_DIR / "latest_model.json"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Height_m",
    "Weight_kg",
    "Age",
    "Chest_in",
    "Length_in",
    "Sleeve_in",
    "ShoulderWidth_in",
]
CATEGORICAL_FEATURES = ["FitPreference"]
# BaseSize is excluded — it's derived from measurements and leaks the target.
TARGET_COLUMN = "SizeClass"

# Ordered so that confusion-matrix axes and adjacent-size checks are meaningful.
SIZE_ORDER = ["S", "M", "L", "XL", "XXL"]

# Adjacent pairs used for the adjacent-size error rate metric.
ADJACENT_PAIRS = {("S", "M"), ("M", "S"), ("M", "L"), ("L", "M"),
                  ("L", "XL"), ("XL", "L"), ("XL", "XXL"), ("XXL", "XL")}

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20          # 80 / 20 train-test split
CV_FOLDS = 5              # Stratified k-fold

# RandomizedSearchCV draws this many candidate combinations.
N_ITER_SEARCH = 30

HYPERPARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

# ---------------------------------------------------------------------------
# Validation bounds for user-supplied measurements
# ---------------------------------------------------------------------------
FEATURE_BOUNDS = {
    "Height_m": (1.30, 2.20),
    "Weight_kg": (30.0, 200.0),
    "Age": (10, 90),
    "Chest_in": (28.0, 60.0),
    "Length_in": (22.0, 40.0),
    "Sleeve_in": (18.0, 36.0),
    "ShoulderWidth_in": (12.0, 26.0),
}
VALID_FIT_PREFERENCES = ["Regular", "Slimfit", "Oversize"]
