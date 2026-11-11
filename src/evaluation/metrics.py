"""Evaluation metrics for pore pressure regression models."""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics:
    - MAE: Mean Absolute Error (bar)
    - MSE: Mean Squared Error (bar^2)
    - RMSE: Root Mean Squared Error (bar)
    - R2: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (%)
    - MaxAE: Maximum Absolute Error (bar)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    valid_mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    if len(yt) == 0:
        return {
            "MAE": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "MaxAE": np.nan,
        }

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 and np.std(yt) > 1e-12 else np.nan
    max_ae = float(np.max(np.abs(yt - yp)))

    # Safe MAPE calculation
    non_zero = yt != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0)
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MaxAE": max_ae,
    }

