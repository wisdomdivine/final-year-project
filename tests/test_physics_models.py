"""Unit tests for physics-based geomechanical models and equations."""

import numpy as np
import pytest

from src.physics.trajectory import compute_minimum_curvature
from src.physics.hydrostatic import calculate_hydrostatic_pressure
from src.physics.overburden import integrate_overburden_stress, calculate_overburden_gradient
from src.physics.eaton import calculate_eaton_pore_pressure
from src.physics.nct import fit_normal_compaction_trend, predict_normal_transit_time


def test_minimum_curvature_vertical_well():
    """Verify that vertical well (inc=0) yields delta TVD equal to delta MD."""
    delta_tvd = compute_minimum_curvature(
        md1=1000.0, inc1_deg=0.0, azi1_deg=0.0,
        md2=1100.0, inc2_deg=0.0, azi2_deg=0.0,
    )
    assert pytest.approx(delta_tvd, rel=1e-5) == 100.0


def test_minimum_curvature_horizontal_well():
    """Verify that horizontal section (inc=90) yields zero delta TVD."""
    delta_tvd = compute_minimum_curvature(
        md1=2000.0, inc1_deg=90.0, azi1_deg=45.0,
        md2=2100.0, inc2_deg=90.0, azi2_deg=45.0,
    )
    assert pytest.approx(delta_tvd, abs=1e-5) == 0.0


def test_hydrostatic_pressure_seabed_and_subsea():
    """Verify hydrostatic reference pressure at seabed and deep subsea."""
    water_depth = 91.0
    seawater_rho = 1.025
    formation_rho = 1.030

    p_seabed = calculate_hydrostatic_pressure(
        tvd_m=water_depth,
        water_depth_m=water_depth,
        seawater_density_gcc=seawater_rho,
        formation_fluid_density_gcc=formation_rho,
    )
    expected_seabed = 1.025 * 0.0980665 * 91.0
    assert pytest.approx(float(p_seabed), rel=1e-4) == expected_seabed

    z_subsea = 2000.0
    p_deep = calculate_hydrostatic_pressure(
        tvd_m=z_subsea,
        water_depth_m=water_depth,
        seawater_density_gcc=seawater_rho,
        formation_fluid_density_gcc=formation_rho,
    )
    expected_deep = expected_seabed + 1.030 * 0.0980665 * (2000.0 - 91.0)
    assert pytest.approx(float(p_deep), rel=1e-4) == expected_deep


def test_overburden_integration_analytical():
    """Verify overburden integration against known constant density slab."""
    water_depth = 91.0
    shallow_base = 262.43
    shallow_rho = 2.011

    tvd = np.array([water_depth, shallow_base, 1000.0])
    rhob = np.array([shallow_rho, shallow_rho, 2.20])

    sv = integrate_overburden_stress(
        tvd_array=tvd,
        rhob_array=rhob,
        water_depth_m=water_depth,
        seawater_density_gcc=1.025,
        shallow_sediment_top_tvd=water_depth,
        shallow_sediment_base_tvd=shallow_base,
        shallow_sediment_density_gcc=shallow_rho,
    )

    p_water = 1.025 * 0.0980665 * 91.0
    p_shallow = p_water + shallow_rho * 0.0980665 * (shallow_base - water_depth)

    assert pytest.approx(sv[0], rel=1e-4) == p_water
    assert pytest.approx(sv[1], rel=1e-4) == p_shallow
    assert sv[2] > sv[1]


def test_eaton_pore_pressure_physics_bounds():
    """Verify physical limits of Eaton's sonic pore pressure formulation."""
    sv = 600.0  # bar
    ph = 300.0  # bar
    dt_norm = 100.0  # us/ft

    # 1. Normally compacted shale (DT_obs == DT_norm): Pp == Ph
    pp_normal = calculate_eaton_pore_pressure(
        sv_bar=sv, ph_bar=ph, dt_observed=dt_norm, dt_normal=dt_norm, exponent=3.0
    )
    assert pytest.approx(float(pp_normal), rel=1e-5) == ph

    # 2. Overpressured shale (DT_obs > DT_norm): Pp > Ph
    dt_undercompacted = 125.0
    pp_overpressure = calculate_eaton_pore_pressure(
        sv_bar=sv, ph_bar=ph, dt_observed=dt_undercompacted, dt_normal=dt_norm, exponent=3.0
    )
    assert float(pp_overpressure) > ph
    assert float(pp_overpressure) < sv

    # 3. Complete undercompaction limit: as DT_obs -> inf, Pp -> Sv
    pp_limit = calculate_eaton_pore_pressure(
        sv_bar=sv, ph_bar=ph, dt_observed=1e6, dt_normal=dt_norm, exponent=3.0
    )
    assert pytest.approx(float(pp_limit), rel=1e-4) == sv


def test_nct_trend_fitting():
    """Verify normal compaction trend parameter fitting and prediction."""
    z = np.linspace(1000, 3000, 20)
    true_dt0 = 200.0
    true_c = 0.0004
    dt_synthetic = true_dt0 * np.exp(-true_c * z)

    fit_results = fit_normal_compaction_trend(z, dt_synthetic)
    assert pytest.approx(fit_results["dt0_us_ft"], rel=1e-3) == true_dt0
    assert pytest.approx(fit_results["compaction_coeff_c"], rel=1e-3) == true_c
    assert pytest.approx(fit_results["r2"], rel=1e-3) == 1.0

    pred_dt = predict_normal_transit_time(z, fit_results["dt0_us_ft"], fit_results["compaction_coeff_c"])
    assert np.allclose(pred_dt, dt_synthetic, rtol=1e-4)

