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


def test_hugin_regression_reconciliation():
    """Verify Hugin oil gradient regression on TVD RKB vs TVD MSL."""
    import pandas as pd
    from pathlib import Path

    p_table = Path("tables/schlumberger_test_point_table.csv")
    assert p_table.exists()
    df = pd.read_csv(p_table)
    hugin = df[(df["run"] == 10) & (df["test_no"] <= 12)]
    assert len(hugin) == 12

    # 1. Regression on tvd_m (RKB datum)
    slope_rkb, intercept_rkb = np.polyfit(hugin["tvd_m"], hugin["formation_pressure_bar"], 1)
    assert pytest.approx(intercept_rkb, abs=1e-3) == 118.205
    assert pytest.approx(slope_rkb, abs=1e-5) == 0.07028

    # 2. Conversion to TVD MSL (TVD_MSL = TVD_RKB - 54.90 m)
    rkb_elev = 54.90
    intercept_msl = intercept_rkb + slope_rkb * rkb_elev
    assert pytest.approx(intercept_msl, abs=1e-3) == 122.063

    # 3. Predict at Test 1 in both coordinate frames
    pred_rkb = intercept_rkb + slope_rkb * 2871.76
    pred_msl = intercept_msl + slope_rkb * (2871.76 - rkb_elev)
    assert pytest.approx(pred_rkb, abs=1e-2) == 320.02
    assert pytest.approx(pred_msl, abs=1e-2) == 320.02


def test_skagerrak_overpressure_reconciliation_trajectory_1():
    """Verify Skagerrak overpressure calculation under corrected Trajectory 1."""
    water_depth = 91.0
    rho_seawater = 1.025
    rho_brine = 1.030

    # At Test 21 (base predictor, TVD MSL = 2983.34 m in Trajectory 1)
    tvd_msl_t21 = 2983.34
    p_meas_t21 = 368.818
    ph_t21 = calculate_hydrostatic_pressure(tvd_msl_t21, water_depth, rho_seawater, rho_brine)
    overpressure_t21 = p_meas_t21 - ph_t21

    assert pytest.approx(float(ph_t21), abs=0.1) == 301.30
    assert pytest.approx(float(overpressure_t21), abs=0.1) == 67.52

    # Deprecated Trajectory 2 check: verify it differed by ~3.76 bar due to the datum shift
    tvd_msl_old_t2 = 3017.83
    ph_old = calculate_hydrostatic_pressure(tvd_msl_old_t2, water_depth, rho_seawater, rho_brine)
    overpressure_old = p_meas_t21 - ph_old
    assert pytest.approx(float(overpressure_old), abs=0.5) == 64.04


def test_skagerrak_apparent_gradient_and_brine_rejection():
    """Verify Skagerrak apparent gradient between Tests 20 and 21 and rejection of brine model."""
    import pandas as pd
    from pathlib import Path

    p_table = Path("tables/schlumberger_test_point_table.csv")
    assert p_table.exists()
    df = pd.read_csv(p_table)

    t20 = df[df["test_no"] == 20].iloc[0]
    t21 = df[df["test_no"] == 21].iloc[0]

    delta_tvd = t21["tvd_m"] - t20["tvd_m"]
    delta_p = t21["formation_pressure_bar"] - t20["formation_pressure_bar"]

    assert pytest.approx(delta_tvd, abs=0.01) == 9.99
    assert pytest.approx(delta_p, abs=0.001) == 0.112

    apparent_gradient = delta_p / delta_tvd
    assert pytest.approx(apparent_gradient, abs=1e-4) == 0.01121

    # Expected brine delta over 9.99 m is ~1.009 bar
    brine_gradient = 0.1010  # bar/m
    expected_brine_delta = delta_tvd * brine_gradient
    assert pytest.approx(expected_brine_delta, abs=0.01) == 1.009

    # Forcing a brine gradient creates a ~0.90 bar error
    brine_forcing_error = abs(expected_brine_delta - delta_p)
    assert pytest.approx(brine_forcing_error, abs=0.01) == 0.897


def test_test_19_memory_recovery_preservation():
    """Verify Test 19 recovered memory pressure preservation and conflict with Test 21."""
    import pandas as pd
    from pathlib import Path

    p_table = Path("tables/schlumberger_test_point_table.csv")
    assert p_table.exists()
    df = pd.read_csv(p_table)

    t19 = df[df["test_no"] == 19].iloc[0]
    t21 = df[df["test_no"] == 21].iloc[0]

    assert pytest.approx(t19["formation_pressure_bar"], abs=0.001) == 357.714
    assert pytest.approx(t21["formation_pressure_bar"], abs=0.001) == 368.818
    assert pytest.approx(t19["drawdown_mobility_md_cp"], abs=0.01) == 5.37

    delta_md = abs(t21["md_m"] - t19["md_m"])
    assert pytest.approx(delta_md, abs=0.01) == 0.04  # 4 cm apart

    conflict_delta_p = t21["formation_pressure_bar"] - t19["formation_pressure_bar"]
    assert pytest.approx(conflict_delta_p, abs=0.001) == 11.104


def test_target_generation_config_policy_integrity():
    """Verify policy guarantees in target generation configuration."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/target_generation.yaml")
    assert cfg_path.exists()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    skag = cfg["formation_regimes"]["Skagerrak"]
    # 1. Tests 20 and 21 preserved as separate measured anchors
    assert skag["primary_measured_anchors"]["test_20"]["formation_pressure_bar"] == 368.706
    assert skag["primary_measured_anchors"]["test_21"]["formation_pressure_bar"] == 368.818
    assert skag["primary_measured_anchors"]["test_20"]["status"] == "measured_anchor"
    assert skag["primary_measured_anchors"]["test_21"]["status"] == "measured_anchor"

    # 2. 368.762 bar is a provisional assumption, not proven physical profile
    assert skag["provisional_compartment_assumption_bar"] == 368.762
    assert skag["provisional_assumption_status"] == "provisional_target_generation_assumption"

    # 3. Gauge repeatability is an unverified assumption
    assert skag["gauge_repeatability_status"] == "unverified_assumption"

    # 4. Test 19 is excluded from calibration
    assert skag["test_19_memory_recovery"]["calibration_role"] == "excluded_from_calibration"
    assert skag["test_19_memory_recovery"]["pressure_bar"] == 357.714

    # 5. Transition zone is strictly excluded from all primary targets
    trans = cfg["transition_zones"]["sleipner_skagerrak"]
    assert trans["primary_policy"] == "exclude_from_primary_target"
    assert trans["mask_primary_training_target"] is True
    assert trans["mask_validation_target"] is True
    assert trans["mask_test_target"] is True
    assert trans["sensitivity_policy"] == "strictly_separate_from_primary_train_val_test"




