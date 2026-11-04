"""Overburden stress (Sv) profile calculation through vertical density integration."""

from typing import Optional, Union
import numpy as np
import pandas as pd


G_FACTOR = 0.0980665  # (g/cm^3) * (m) * (9.80665 m/s^2) / 100 -> bar


def integrate_overburden_stress(
    tvd_array: np.ndarray,
    rhob_array: np.ndarray,
    water_depth_m: float = 91.0,
    seawater_density_gcc: float = 1.025,
    shallow_sediment_top_tvd: float = 91.0,
    shallow_sediment_base_tvd: float = 262.43,
    shallow_sediment_density_gcc: float = 2.011,
) -> np.ndarray:
    """
    Compute total vertical overburden stress Sv (bar) along TVD (m MSL).
    
    Sv(z) = P_water + P_shallow + integral(rhob * g * dz)
    """
    # Water column pressure at seabed
    p_water = seawater_density_gcc * G_FACTOR * water_depth_m

    # Shallow sediment interval pressure
    delta_shallow = max(0.0, shallow_sediment_base_tvd - shallow_sediment_top_tvd)
    p_shallow = shallow_sediment_density_gcc * G_FACTOR * delta_shallow

    p_base_shallow = p_water + p_shallow

    # Ensure sorted by TVD
    sort_idx = np.argsort(tvd_array)
    tvd_sorted = tvd_array[sort_idx]
    rhob_sorted = rhob_array[sort_idx]

    sv_sorted = np.zeros_like(tvd_sorted, dtype=float)

    # For points above or at base of shallow sediment
    mask_shallow = tvd_sorted <= shallow_sediment_base_tvd
    for i in np.where(mask_shallow)[0]:
        z = tvd_sorted[i]
        if z <= water_depth_m:
            sv_sorted[i] = seawater_density_gcc * G_FACTOR * z
        else:
            sv_sorted[i] = p_water + shallow_sediment_density_gcc * G_FACTOR * (z - water_depth_m)

    # For points below shallow sediment, integrate log density
    mask_deep = tvd_sorted > shallow_sediment_base_tvd
    deep_indices = np.where(mask_deep)[0]

    if len(deep_indices) > 0:
        first_deep_idx = deep_indices[0]
        # Step from base of shallow sediment to first deep point
        z_prev = shallow_sediment_base_tvd
        sv_acc = p_base_shallow

        for i in deep_indices:
            z_curr = tvd_sorted[i]
            dz = z_curr - z_prev
            rho_avg = rhob_sorted[i]
            sv_acc += rho_avg * G_FACTOR * dz
            sv_sorted[i] = sv_acc
            z_prev = z_curr

    # Invert sorting to match original order
    sv_out = np.empty_like(sv_sorted)
    sv_out[sort_idx] = sv_sorted
    return sv_out


def calculate_overburden_gradient(
    sv_bar: np.ndarray,
    tvd_m: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate overburden gradient in bar/m, psi/ft, and equivalent specific gravity (s.g.).
    """
    # Safe division avoiding divide by zero at surface
    safe_tvd = np.where(tvd_m > 0, tvd_m, np.nan)
    obg_bar_m = sv_bar / safe_tvd
    obg_sg = obg_bar_m / G_FACTOR
    obg_psi_ft = obg_bar_m * (14.50377 / 3.28084)

    return pd.DataFrame({
        "TVD_MSL": tvd_m,
        "Sv_bar": sv_bar,
        "OBG_bar_m": obg_bar_m,
        "OBG_sg": obg_sg,
        "OBG_psi_ft": obg_psi_ft,
    })
