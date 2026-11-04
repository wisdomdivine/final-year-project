"""Residual analysis and diagnostic statistics."""

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_residuals(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Calculate model residuals: residuals = y_true - y_pred."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return yt - yp


def analyze_residuals(residuals: np.ndarray) -> Dict[str, float]:
    """
    Compute distribution properties of residuals (mean, std, skewness, kurtosis, quantiles).
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]

    if len(res) == 0:
        return {}

    mean_res = float(np.mean(res))
    std_res = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
    median_res = float(np.median(res))
    q25 = float(np.percentile(res, 25))
    q75 = float(np.percentile(res, 75))
    iqr = q75 - q25

    return {
        "mean_residual": mean_res,
        "std_residual": std_res,
        "median_residual": median_res,
        "q25_residual": q25,
        "q75_residual": q75,
        "iqr_residual": iqr,
        "min_residual": float(np.min(res)),
        "max_residual": float(np.max(res)),
    }

