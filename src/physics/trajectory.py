"""Wellbore directional trajectory and TVD calculation."""

from typing import Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate_tvd_from_survey(
    md_query: Union[float, np.ndarray, pd.Series],
    survey_df: pd.DataFrame,
    md_col: str = "MD",
    tvd_col: str = "TVD MSL",
) -> np.ndarray:
    """
    Interpolate true vertical depth (TVD MSL) from directional survey table using linear interpolation.
    """
    survey_sorted = survey_df.sort_values(by=md_col).dropna(subset=[md_col, tvd_col])
    md_points = survey_sorted[md_col].values
    tvd_points = survey_sorted[tvd_col].values

    interpolator = interp1d(
        md_points,
        tvd_points,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(interpolator(md_query), dtype=float)


def compute_minimum_curvature(
    md1: float,
    inc1_deg: float,
    azi1_deg: float,
    md2: float,
    inc2_deg: float,
    azi2_deg: float,
) -> float:
    """
    Calculate incremental TVD between two survey stations using the Minimum Curvature Method.
    """
    delta_md = md2 - md1
    if delta_md == 0:
        return 0.0

    i1 = np.radians(inc1_deg)
    i2 = np.radians(inc2_deg)
    a1 = np.radians(azi1_deg)
    a2 = np.radians(azi2_deg)

    # Dogleg angle calculation
    cos_dogleg = np.cos(i2 - i1) - (np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1)))
    cos_dogleg = np.clip(cos_dogleg, -1.0, 1.0)
    dogleg = np.arccos(cos_dogleg)

    # Ratio factor
    if abs(dogleg) < 1e-6:
        rf = 1.0
    else:
        rf = (2.0 / dogleg) * np.tan(dogleg / 2.0)

    delta_tvd = (delta_md / 2.0) * (np.cos(i1) + np.cos(i2)) * rf
    return float(delta_tvd)
