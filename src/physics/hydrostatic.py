"""Hydrostatic normal pressure reference calculation."""

from typing import Union
import numpy as np
import pandas as pd


G_FACTOR = 0.0980665


def calculate_hydrostatic_pressure(
    tvd_m: Union[float, np.ndarray, pd.Series],
    water_depth_m: float = 91.0,
    seawater_density_gcc: float = 1.025,
    formation_fluid_density_gcc: float = 1.030,
) -> Union[float, np.ndarray]:
    """
    Calculate normal hydrostatic reference pressure (bar) at given TVD (m MSL).
    
    Ph(z) = P_seabed + rho_f * g * (z - water_depth_m)
    """
    is_scalar = np.isscalar(tvd_m)
    tvd_arr = np.atleast_1d(np.asarray(tvd_m, dtype=float))
    p_seabed = seawater_density_gcc * G_FACTOR * water_depth_m

    # Below seabed
    subsea_depth = np.maximum(0.0, tvd_arr - water_depth_m)
    ph = p_seabed + (formation_fluid_density_gcc * G_FACTOR * subsea_depth)

    # For points inside water column (z <= water_depth_m)
    in_water = tvd_arr <= water_depth_m
    ph[in_water] = seawater_density_gcc * G_FACTOR * tvd_arr[in_water]

    if is_scalar:
        return float(ph[0])
    return ph
