"""Feature transformation, scaling, and contiguous depth partitioning."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


DEFAULT_SPLIT_POINTS = {
    "train_end": 3320.7960,
    "val_end": 3362.2488,
    "test_end": 3403.5492,
}

LOG_TRANSFORM_COLUMNS = ["K", "RS", "RD", "RT"]


def split_dataset_by_depth(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    train_end: float = DEFAULT_SPLIT_POINTS["train_end"],
    val_end: float = DEFAULT_SPLIT_POINTS["val_end"],
    test_end: float = DEFAULT_SPLIT_POINTS["test_end"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically/contiguously along depth without shuffling.
    Prevents spatial data leakage.
    """
    df_sorted = df.sort_values(by=depth_col).reset_index(drop=True)

    train_mask = df_sorted[depth_col] <= train_end
    val_mask = (df_sorted[depth_col] > train_end) & (df_sorted[depth_col] <= val_end)
    test_mask = (df_sorted[depth_col] > val_end) & (df_sorted[depth_col] <= test_end)

    df_train = df_sorted[train_mask].copy().reset_index(drop=True)
    df_val = df_sorted[val_mask].copy().reset_index(drop=True)
    df_test = df_sorted[test_mask].copy().reset_index(drop=True)

    return df_train, df_val, df_test


def transform_log_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    epsilon: float = 1e-4,
) -> pd.DataFrame:
    """
    Apply log10 transformation to skewed resistivity and compositional features.
    Clips values at epsilon to handle non-positive measurements robustly.
    """
    if columns is None:
        columns = LOG_TRANSFORM_COLUMNS

    df_out = df.copy()
    for col in columns:
        if col in df_out.columns:
            clipped = np.clip(df_out[col].values, a_min=epsilon, a_max=None)
            df_out[f"{col}_log10"] = np.log10(clipped)

    return df_out


def scale_dataset_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """
    Fit scaler exclusively on df_train and transform all splits.
    
    scaler_type options: 'standard', 'minmax', 'robust'.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    train_scaled = df_train.copy()
    val_scaled = df_val.copy()
    test_scaled = df_test.copy()

    scaler.fit(df_train[feature_columns])

    train_scaled[feature_columns] = scaler.transform(df_train[feature_columns])
    val_scaled[feature_columns] = scaler.transform(df_val[feature_columns])
    test_scaled[feature_columns] = scaler.transform(df_test[feature_columns])

    return train_scaled, val_scaled, test_scaled, scaler
