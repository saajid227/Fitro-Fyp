"""Data loading and validation for the Fitaro pipeline.

Responsibilities:
- Read the CSV from the configured path.
- Confirm the schema matches expectations.
- Flag (and optionally drop) missing or out-of-range values.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    FEATURE_BOUNDS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DataLoadError(Exception):
    """Raised when the CSV cannot be read or is structurally empty."""


class SchemaValidationError(Exception):
    """Raised when required columns are missing from the dataset."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the garment-size dataset.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        A clean DataFrame containing only the columns needed for modelling.

    Raises:
        DataLoadError: If the file cannot be read or is empty.
        SchemaValidationError: If required columns are missing.
    """
    logger.info("Loading dataset from %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Dataset not found at {path}") from exc
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    if df.empty:
        raise DataLoadError("Dataset is empty after loading.")

    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))

    _validate_schema(df)
    df = _handle_missing_values(df)
    df = _validate_numeric_ranges(df)

    # Drop BaseSize — it encodes size information derived purely from
    # measurements, so keeping it would leak the target during inference.
    if "BaseSize" in df.columns:
        df = df.drop(columns=["BaseSize"])
        logger.debug("Dropped 'BaseSize' column to prevent target leakage.")

    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    return df[required_cols]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaValidationError if any expected column is absent."""
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )
    logger.debug("Schema validation passed.")


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Log warnings for missing values and drop affected rows."""
    missing_counts = df.isnull().sum()
    cols_with_na = missing_counts[missing_counts > 0]

    if not cols_with_na.empty:
        for col, count in cols_with_na.items():
            logger.warning("Column '%s' has %d missing value(s).", col, count)
        rows_before = len(df)
        df = df.dropna()
        logger.warning(
            "Dropped %d row(s) containing missing values.", rows_before - len(df)
        )
    else:
        logger.debug("No missing values found.")

    return df


def _validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Warn and remove rows where numeric features fall outside expected bounds."""
    mask_valid = pd.Series(True, index=df.index)

    for col, (lo, hi) in FEATURE_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = ~df[col].between(lo, hi)
        n_bad = out_of_range.sum()
        if n_bad:
            logger.warning(
                "Column '%s': %d value(s) outside [%s, %s] — will be removed.",
                col, n_bad, lo, hi,
            )
        mask_valid &= ~out_of_range

    rows_before = len(df)
    df = df[mask_valid]
    removed = rows_before - len(df)
    if removed:
        logger.warning("Removed %d row(s) with out-of-range numeric values.", removed)

    return df
