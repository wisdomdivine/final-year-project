"""Eaton method for pore pressure estimation from acoustic log response."""

from typing import Union
import numpy as np


def calculate_eaton_pore_pressure(
    sv_bar: Union[float, np.ndarray],
    ph_bar: Union[float, np.ndarray],
    dt_observed: Union[float, np.ndarray],
    dt_normal: Union[float, np.ndarray],
    exponent: float = 3.0,
) -> Union[float, np.ndarray]:
    """
    Compute pore pressure using Eaton's sonic equation:
    
    Pp = Sv - (Sv - Ph) * (DT_normal / DT_observed) ** exponent
    
    Parameters
    ----------
    sv_bar : float or array-like
        Total vertical overburden stress (bar).
    ph_bar : float or array-like
        Normal hydrostatic pressure (bar).
    dt_observed : float or array-like
        Observed sonic interval transit time (us/ft).
    dt_normal : float or array-like
        Normal compaction trend sonic transit time (us/ft).
    exponent : float, default 3.0
        Eaton sonic exponent (traditionally 3.0 in Gulf of Mexico shales).
        
    Returns
    -------
    pp_bar : float or array-like
        Predicted formation pore pressure (bar).
    """
    sv = np.asarray(sv_bar, dtype=float)
    ph = np.asarray(ph_bar, dtype=float)
    dt_obs = np.asarray(dt_observed, dtype=float)
    dt_norm = np.asarray(dt_normal, dtype=float)

    # Avoid zero division
    ratio = np.divide(dt_norm, dt_obs, out=np.zeros_like(dt_norm), where=dt_obs != 0)
    ratio_powered = np.power(ratio, exponent)

    effective_vertical_stress_normal = sv - ph
    effective_vertical_stress_actual = effective_vertical_stress_normal * ratio_powered

    pp = sv - effective_vertical_stress_actual
    return pp
