"""Data loading utilities for Well 15/9-F-12 datasets."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_predictor_dataset(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the 1,811-observation predictor dataset."""
    if path is None:
        path = PROJECT_ROOT / "dataset" / "F12_predictor_dataset_1811.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Predictor dataset not found at {p}")
    return pd.read_csv(p)


def load_processed_splits(
    processed_dir: Optional[Union[str, Path]] = None,
    variant: str = "raw",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test feature matrices.
    
    variant can be 'raw', 'scaled', or 'log_scaled'.
    """
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "dataset" / "processed"
    p = Path(processed_dir)

    train_file = p / f"X_train_{variant}.csv"
    val_file = p / f"X_val_{variant}.csv"
    test_file = p / f"X_test_{variant}.csv"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        raise FileNotFoundError(f"One or more split files for variant '{variant}' not found in {p}")

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    return df_train, df_val, df_test


def load_directional_survey(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the official directional survey for 15/9-F-12."""
    if path is None:
        path = PROJECT_ROOT / "data-raw" / "Well_15_9-F-12_Dir_survey.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Directional survey not found at {p}")
    return pd.read_csv(p)


def load_formation_pressure_tests(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the reconciled formation pressure tests table."""
    if path is None:
        path = PROJECT_ROOT / "tables" / "mwd_formation_pressure_tests.csv"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Formation pressure tests table not found at {p}")
    return pd.read_csv(p)

