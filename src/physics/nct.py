"""Normal Compaction Trend (NCT) fitting for sonic log response in shale."""

from typing import Dict, Tuple
import numpy as np


def fit_normal_compaction_trend(
    tvd_m: np.ndarray,
    dt_us_ft: np.ndarray,
) -> Dict[str, float]:
    """
    Fit an exponential Normal Compaction Trend:
    ln(DT) = ln(DT0) - c * TVD  =>  DT_n(z) = DT0 * exp(-c * z)
    
    Returns parameters DT0 and c, along with R-squared.
    """
    valid_mask = (tvd_m > 0) & (dt_us_ft > 0) & np.isfinite(tvd_m) & np.isfinite(dt_us_ft)
    z = tvd_m[valid_mask]
    dt = dt_us_ft[valid_mask]

    if len(z) < 2:
        raise ValueError("Insufficient data points to fit normal compaction trend.")

    log_dt = np.log(dt)
    slope, intercept = np.polyfit(z, log_dt, 1)

    c = -slope
    dt0 = float(np.exp(intercept))

    pred_log_dt = intercept + slope * z
    ss_res = np.sum((log_dt - pred_log_dt) ** 2)
    ss_tot = np.sum((log_dt - np.mean(log_dt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "dt0_us_ft": dt0,
        "compaction_coeff_c": float(c),
        "r2": r2,
    }


def predict_normal_transit_time(
    tvd_m: np.ndarray,
    dt0_us_ft: float,
    compaction_coeff_c: float,
) -> np.ndarray:
    """Predict normal compaction sonic transit time DT_n given depth."""
    return dt0_us_ft * np.exp(-compaction_coeff_c * tvd_m)
