"""
Formation Pore-Pressure Target Vector Generation and Provenance Tracking
for Well 15/9-F-12 (Volve Field).

This module implements the physically constrained target generation methodology
reconciled during Phase 2, strictly enforcing:
1. Provenance distinction across every depth observation:
   - 'measured_anchor': exact grid points anchored to stabilized physical pretests.
   - 'physics_derived': continuous hydrostatic fluid columns (oil in Hugin, brine in Sleipner).
   - 'provisional_assumption': provisional constant compartment pressure in Skagerrak.
   - 'invalid_unsupported': unmeasured interbedded transition zone (3,326.0 to 3,386.5 m MD).
2. Strict split separation:
   - Tests 20 & 21 reside exclusively in the Test partition (MD 3362.40 - 3403.55 m).
   - Zero influence of Test partition stations on Training or Validation.
3. Secondary diagnostic sensitivity curves:
   - Linear ramp across transition zone.
   - Sharp top Skagerrak seal at 3,340.0 m MD.
   - Sharp basal Sleipner seal at 3,326.3 m MD.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def generate_target_vectors(
    predictor_df: Optional[pd.DataFrame] = None,
    trajectory_df: Optional[pd.DataFrame] = None,
    pressure_rec_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate the complete 1,811-row target dataset with explicit provenance."""
    base_dir = Path(__file__).resolve().parents[2]

    if predictor_df is None:
        predictor_df = pd.read_csv(base_dir / "dataset" / "F12_predictor_dataset_1811.csv")
    if trajectory_df is None:
        trajectory_df = pd.read_csv(base_dir / "outputs" / "audit" / "trajectory_reconciliation.csv")
    if pressure_rec_df is None:
        pressure_rec_df = pd.read_csv(base_dir / "outputs" / "audit" / "pressure_station_reconciliation.csv")

    n_obs = len(predictor_df)
    assert n_obs == 1811, f"Expected 1811 rows, got {n_obs}"
    assert len(trajectory_df) == 1811, f"Trajectory rows mismatch: {len(trajectory_df)}"

    depths = predictor_df["DEPTH"].values
    tvd_msl = trajectory_df["TVD_MSL_traj1_sensitivity_m"].values
    tvd_rkb = trajectory_df["TVD_RKB_traj1_sensitivity_m"].values

    # Pretest anchor stations lookup
    # Only Run 10 tests within predictor range
    p_r10 = pressure_rec_df[pressure_rec_df["xls_md_m"].between(depths[0] - 0.5, depths[-1] + 0.5)].copy()

    # Pre-identify anchor grid indices for verified stations
    # Hugin: Tests 1 to 12
    # Sleipner: Tests 14, 15, 16 (Test 13 is aborted pretest; Test 17 is dry test)
    # Skagerrak: Tests 20, 21 (Test 18 is fair/aborted; Test 19 is unverified memory conflict)
    anchor_tests = {
        1: (3127.71, 320.020),
        2: (3127.99, 320.042),
        3: (3136.74, 320.385),
        4: (3144.07, 320.713),
        5: (3152.61, 321.050),
        6: (3170.73, 321.827),
        7: (3191.85, 322.677),
        8: (3221.68, 323.892),
        9: (3240.73, 324.698),
        10: (3254.85, 325.298),
        11: (3263.51, 325.702),
        12: (3262.78, 325.686),
        14: (3294.00, 334.101),
        15: (3307.73, 335.610),
        16: (3324.65, 341.713),
        20: (3386.99, 368.706),
        21: (3403.57, 368.818),
    }

    anchor_grid_map: Dict[int, Tuple[int, float, float]] = {}
    for t_no, (t_md, t_p) in anchor_tests.items():
        c_idx = int(np.argmin(np.abs(depths - t_md)))
        anchor_grid_map[c_idx] = (t_no, t_md, t_p)

    # Initialize output target arrays
    formation = np.empty(n_obs, dtype=object)
    is_target_valid = np.zeros(n_obs, dtype=bool)
    target_provenance = np.empty(n_obs, dtype=object)
    pp_target_bar = np.full(n_obs, np.nan, dtype=float)
    pp_target_uncertainty_bar = np.full(n_obs, np.nan, dtype=float)
    measured_test_no = [None] * n_obs
    measured_pressure_bar = [None] * n_obs
    partition = np.empty(n_obs, dtype=object)
    derivation_method = np.empty(n_obs, dtype=object)

    # Sensitivity targets
    pp_sens_ramp = np.full(n_obs, np.nan, dtype=float)
    pp_sens_topseal = np.full(n_obs, np.nan, dtype=float)
    pp_sens_basalseal = np.full(n_obs, np.nan, dtype=float)

    # Formation and compartment references
    # Hugin: oil gradient 0.070276 bar/m anchored to Test 1 (index 0)
    t1_tvd = tvd_msl[0]
    t1_p = 320.020
    hugin_oil_grad = 0.070276

    # Sleipner: formation brine gradient 0.1010 bar/m
    brine_grad = 0.1010
    idx_t14 = int(np.argmin(np.abs(depths - 3294.00)))
    idx_t15 = int(np.argmin(np.abs(depths - 3307.73)))
    idx_t16 = int(np.argmin(np.abs(depths - 3324.65)))

    t14_tvd = tvd_msl[idx_t14]
    t14_p = 334.101
    t15_tvd = tvd_msl[idx_t15]
    t15_p = 335.610
    t16_tvd = tvd_msl[idx_t16]
    t16_p = 341.713

    # Skagerrak provisional constant compartment assumption
    skag_provisional_p = 368.762

    # Loop through all observations
    for i in range(n_obs):
        z_md = depths[i]
        z_tvd = tvd_msl[i]

        # 1. Assign partition (70:15:15 depth split)
        if i < 1268:
            part = "Training"
        elif i < 1268 + 272:
            part = "Validation"
        else:
            part = "Testing"
        partition[i] = part

        # 2. Check if this grid node is a discrete measured anchor
        is_anchor = i in anchor_grid_map

        # 3. Formation-specific target formulation
        if z_md <= 3263.50:
            # --- Hugin Formation ---
            formation[i] = "Hugin"
            is_target_valid[i] = True
            pp_target_uncertainty_bar[i] = 0.05

            if is_anchor:
                t_no, _, t_p = anchor_grid_map[i]
                target_provenance[i] = "measured_anchor"
                measured_test_no[i] = int(t_no)
                measured_pressure_bar[i] = float(t_p)
                pp_target_bar[i] = float(t_p)
                derivation_method[i] = "measured_pretest_anchor"
            else:
                target_provenance[i] = "physics_derived"
                pp_target_bar[i] = t1_p + hugin_oil_grad * (z_tvd - t1_tvd)
                derivation_method[i] = "hugin_oil_hydrostatic_regression"

            pp_sens_ramp[i] = pp_target_bar[i]
            pp_sens_topseal[i] = pp_target_bar[i]
            pp_sens_basalseal[i] = pp_target_bar[i]

        elif z_md <= 3326.00:
            # --- Sleipner Formation (Upper/Middle/Lower sands) ---
            formation[i] = "Sleipner"
            is_target_valid[i] = True

            if is_anchor:
                t_no, _, t_p = anchor_grid_map[i]
                target_provenance[i] = "measured_anchor"
                measured_test_no[i] = int(t_no)
                measured_pressure_bar[i] = float(t_p)
                pp_target_bar[i] = float(t_p)
                pp_target_uncertainty_bar[i] = 0.20
                derivation_method[i] = "measured_pretest_anchor"
            else:
                target_provenance[i] = "physics_derived"
                if z_md <= 3296.80:
                    # Upper Sand compartment anchored to Test 14
                    pp_target_bar[i] = t14_p + brine_grad * (z_tvd - t14_tvd)
                    pp_target_uncertainty_bar[i] = 0.50
                    derivation_method[i] = "sleipner_upper_sand_brine_gradient"
                elif z_md <= 3315.00:
                    # Middle Sand compartment anchored to Test 15
                    pp_target_bar[i] = t15_p + brine_grad * (z_tvd - t15_tvd)
                    pp_target_uncertainty_bar[i] = 0.50
                    derivation_method[i] = "sleipner_middle_sand_brine_gradient"
                else:
                    # Lower Sand compartment anchored to Test 16
                    pp_target_bar[i] = t16_p + brine_grad * (z_tvd - t16_tvd)
                    pp_target_uncertainty_bar[i] = 0.80
                    derivation_method[i] = "sleipner_lower_sand_brine_gradient"

            pp_sens_ramp[i] = pp_target_bar[i]
            pp_sens_topseal[i] = pp_target_bar[i]
            pp_sens_basalseal[i] = pp_target_bar[i]

        elif z_md <= 3386.50:
            # --- Sleipner-Skagerrak Transition Zone ---
            # Unmeasured interbedded succession (76% shale/mudstone, 24% silty sand)
            # Strictly excluded from primary training, validation, and test targets
            formation[i] = "Transition_Zone"
            is_target_valid[i] = False
            target_provenance[i] = "invalid_unsupported"
            pp_target_bar[i] = np.nan
            pp_target_uncertainty_bar[i] = np.nan
            derivation_method[i] = "none_unmeasured_transition"

            # Sensitivity Scenario 1: Continuous linear ramp
            # Ramp connects lower Sleipner base (341.713 bar) to Skagerrak anchor (368.706 bar)
            frac = (z_md - 3326.0) / (3386.5 - 3326.0)
            pp_sens_ramp[i] = 341.713 + frac * (368.706 - 341.713)

            # Sensitivity Scenario 2: Sharp top Skagerrak seal at 3,340.0 m MD
            if z_md < 3340.0:
                pp_sens_topseal[i] = t16_p + brine_grad * (z_tvd - t16_tvd)
            else:
                pp_sens_topseal[i] = 368.706

            # Sensitivity Scenario 3: Sharp basal hardground seal at 3,326.3 m MD
            if z_md < 3326.3:
                pp_sens_basalseal[i] = t16_p + brine_grad * (z_tvd - t16_tvd)
            else:
                pp_sens_basalseal[i] = 368.706

        else:
            # --- Skagerrak Formation Reservoir Interval ---
            formation[i] = "Skagerrak"
            is_target_valid[i] = True

            if is_anchor:
                t_no, _, t_p = anchor_grid_map[i]
                target_provenance[i] = "measured_anchor"
                measured_test_no[i] = int(t_no)
                measured_pressure_bar[i] = float(t_p)
                pp_target_bar[i] = float(t_p)
                pp_target_uncertainty_bar[i] = 0.15
                derivation_method[i] = f"measured_pretest_anchor (Test {t_no})"
            else:
                # Provisional target-generation assumption (explicitly NOT a proven physical profile)
                target_provenance[i] = "provisional_assumption"
                pp_target_bar[i] = skag_provisional_p
                pp_target_uncertainty_bar[i] = 0.15
                derivation_method[i] = "skagerrak_provisional_constant_compartment"

            pp_sens_ramp[i] = pp_target_bar[i]
            pp_sens_topseal[i] = pp_target_bar[i]
            pp_sens_basalseal[i] = pp_target_bar[i]

    targets_df = pd.DataFrame({
        "observation_index": np.arange(n_obs),
        "DEPTH": depths,
        "TVD_MSL": tvd_msl,
        "TVD_RKB": tvd_rkb,
        "FORMATION": formation,
        "PARTITION": partition,
        "IS_TARGET_VALID": is_target_valid,
        "TARGET_PROVENANCE": target_provenance,
        "PP_TARGET_BAR": pp_target_bar,
        "PP_TARGET_UNCERTAINTY_BAR": pp_target_uncertainty_bar,
        "MEASURED_TEST_NO": measured_test_no,
        "MEASURED_PRESSURE_BAR": measured_pressure_bar,
        "DERIVATION_METHOD": derivation_method,
        "PP_SENSITIVITY_RAMP_BAR": pp_sens_ramp,
        "PP_SENSITIVITY_TOPSEAL_BAR": pp_sens_topseal,
        "PP_SENSITIVITY_BASALSEAL_BAR": pp_sens_basalseal,
    })

    return targets_df


def audit_target_generation(
    targets_df: pd.DataFrame,
    pressure_rec_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Audit the generated target dataset against physical stations and requirements."""
    base_dir = Path(__file__).resolve().parents[2]
    if pressure_rec_df is None:
        pressure_rec_df = pd.read_csv(base_dir / "outputs" / "audit" / "pressure_station_reconciliation.csv")

    total_obs = len(targets_df)
    valid_obs = int(targets_df["IS_TARGET_VALID"].sum())
    invalid_obs = total_obs - valid_obs

    # Provenance counts
    prov_counts = targets_df["TARGET_PROVENANCE"].value_counts().to_dict()

    # Partition counts
    part_summary = {}
    for part in ["Training", "Validation", "Testing"]:
        sub = targets_df[targets_df["PARTITION"] == part]
        part_summary[part] = {
            "total_rows": len(sub),
            "valid_rows": int(sub["IS_TARGET_VALID"].sum()),
            "invalid_rows": int((~sub["IS_TARGET_VALID"]).sum()),
            "depth_range_md": [round(float(sub["DEPTH"].min()), 4), round(float(sub["DEPTH"].max()), 4)],
            "provenance_breakdown": sub["TARGET_PROVENANCE"].value_counts().to_dict(),
        }

    # Formation coverage and pressure ranges
    formation_summary = {}
    for fmt in ["Hugin", "Sleipner", "Transition_Zone", "Skagerrak"]:
        sub = targets_df[targets_df["FORMATION"] == fmt]
        valid_sub = sub[sub["IS_TARGET_VALID"]]
        p_vals = valid_sub["PP_TARGET_BAR"].dropna()

        formation_summary[fmt] = {
            "total_rows": len(sub),
            "valid_rows": len(valid_sub),
            "depth_range_md": [round(float(sub["DEPTH"].min()), 4), round(float(sub["DEPTH"].max()), 4)],
            "tvd_msl_range": [round(float(sub["TVD_MSL"].min()), 2), round(float(sub["TVD_MSL"].max()), 2)],
            "tvd_rkb_range": [round(float(sub["TVD_RKB"].min()), 2), round(float(sub["TVD_RKB"].max()), 2)],
            "pressure_min_bar": round(float(p_vals.min()), 3) if len(p_vals) > 0 else None,
            "pressure_max_bar": round(float(p_vals.max()), 3) if len(p_vals) > 0 else None,
            "pressure_mean_bar": round(float(p_vals.mean()), 3) if len(p_vals) > 0 else None,
            "pressure_std_bar": round(float(p_vals.std()), 3) if len(p_vals) > 0 else None,
        }

    # Missing target location analysis (Transition Zone)
    missing_sub = targets_df[~targets_df["IS_TARGET_VALID"]]
    missing_summary = {
        "interval_md": [round(float(missing_sub["DEPTH"].min()), 4), round(float(missing_sub["DEPTH"].max()), 4)],
        "row_count": len(missing_sub),
        "status": "excluded_from_primary_targets",
        "partitions_spanned": missing_sub["PARTITION"].unique().tolist(),
        "dry_test_station": "Test 17 (3,333.57 m MD, 69.279 bar in tight formation)",
        "lithology_breakdown": {
            "shale_mudstone_pct": 76.07,
            "silty_sandstone_pct": 23.93,
            "note": "Interbedded succession of silty shales, claystones, and thin silty sandstone stringers; not pure mudstone.",
        },
    }

    # Station-by-station comparison for all Run 10 tests
    station_comparisons = []
    p_r10 = pressure_rec_df[pressure_rec_df["xls_md_m"].between(targets_df["DEPTH"].min() - 0.5, targets_df["DEPTH"].max() + 0.5)].copy()

    for _, row in p_r10.iterrows():
        t_no = int(row["test_no"])
        t_md = float(row["xls_md_m"])
        t_table_p = float(row["xls_formation_pressure_bar"])
        t_bu_p = float(row["xls_last_read_buildup_bar"]) if pd.notna(row["xls_last_read_buildup_bar"]) else t_table_p
        t_rem = str(row["xls_remarks"])

        c_idx = int(np.argmin(np.abs(targets_df["DEPTH"].values - t_md)))
        grid_row = targets_df.iloc[c_idx]

        grid_md = float(grid_row["DEPTH"])
        grid_p = grid_row["PP_TARGET_BAR"]
        grid_prov = str(grid_row["TARGET_PROVENANCE"])
        grid_part = str(grid_row["PARTITION"])
        grid_source_test = grid_row["MEASURED_TEST_NO"]
        md_dist = round(abs(grid_md - t_md), 4)

        # 1. Reconcile Test 13: FPWD pressure-versus-time plot reached 334.161 bar buildup.
        # Table entry 328.056 bar reflects investigation drawdown stage.
        if t_no == 13:
            reconciled_p = 334.161  # From FPWD plot p. 134
            conflicting_table_p = 328.056
            diff_p = round(grid_p - reconciled_p, 3) if pd.notna(grid_p) else None
            station_comparisons.append({
                "test_no": t_no,
                "station_role": "aborted_pretest_reconciled",
                "station_md": t_md,
                "reconciled_station_p_bar": reconciled_p,
                "conflicting_table_p_bar": conflicting_table_p,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": round(float(grid_p), 3) if pd.notna(grid_p) else None,
                "target_residual_bar": diff_p,
                "target_source_test_no": None,
                "target_provenance": grid_prov,
                "station_provenance": "reconciled_buildup_plot",
                "partition": grid_part,
                "remarks": "FPWD plot reached 334.161 bar buildup; table 328.056 bar was investigation drawdown; companion Test 14 anchored at 334.101 bar.",
                "status": "reconciled_via_plot",
            })

        # 2. Reconcile Test 19 vs Test 21: Grid node 1810 is strictly sourced from Test 21.
        # Test 19 memory recovery (357.714 bar) is an alternative conflict observation, excluded from anchors.
        elif t_no == 19:
            diff_p = round(grid_p - t_table_p, 3) if pd.notna(grid_p) else None
            station_comparisons.append({
                "test_no": t_no,
                "station_role": "alternative_memory_observation",
                "station_md": t_md,
                "reconciled_station_p_bar": t_table_p,
                "conflicting_table_p_bar": None,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": round(float(grid_p), 3) if pd.notna(grid_p) else None,
                "target_residual_bar": diff_p,
                "target_source_test_no": 21,  # Target is strictly sourced from companion Test 21, NOT Test 19
                "target_provenance": grid_prov,
                "station_provenance": "uncalibrated_memory_conflict",
                "partition": grid_part,
                "remarks": "Tool telemetry shutdown; post-run memory recovery 357.714 bar; excluded from primary calibration; grid target originates exclusively from companion Test 21.",
                "status": "unresolved_memory_conflict",
            })

        elif t_no == 21:
            diff_p = round(grid_p - t_table_p, 3) if pd.notna(grid_p) else None
            station_comparisons.append({
                "test_no": t_no,
                "station_role": "independent_test_evaluation_station",
                "station_md": t_md,
                "reconciled_station_p_bar": t_table_p,
                "conflicting_table_p_bar": None,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": round(float(grid_p), 3) if pd.notna(grid_p) else None,
                "target_residual_bar": diff_p,
                "target_source_test_no": 21,
                "target_provenance": grid_prov,
                "station_provenance": "measured_anchor",
                "partition": grid_part,
                "remarks": "Primary stabilized reference for Skagerrak reservoir sand; companion repeat for Test 19; source of grid target 368.818 bar.",
                "status": "matched",
            })

        elif t_no == 17:
            station_comparisons.append({
                "test_no": t_no,
                "station_role": "dry_test_excluded",
                "station_md": t_md,
                "reconciled_station_p_bar": t_table_p,
                "conflicting_table_p_bar": None,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": None,
                "target_residual_bar": None,
                "target_source_test_no": None,
                "target_provenance": grid_prov,
                "station_provenance": "unsupported_tight_facies",
                "partition": grid_part,
                "remarks": "Dry test in tight formation (69.279 bar); located in unsupported transition zone; excluded from quantitative targets.",
                "status": "unsupported_transition",
            })

        elif t_no == 18:
            station_comparisons.append({
                "test_no": t_no,
                "station_role": "initial_pretest_fair_quality",
                "station_md": t_md,
                "reconciled_station_p_bar": t_table_p,
                "conflicting_table_p_bar": None,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": None,
                "target_residual_bar": None,
                "target_source_test_no": None,
                "target_provenance": grid_prov,
                "station_provenance": "unsupported_transition_boundary",
                "partition": grid_part,
                "remarks": "Initial incomplete pretest (354.001 bar); located in transition zone boundary; companion repeat Test 20 stabilized at 368.706 bar.",
                "status": "unsupported_transition",
            })

        else:
            diff_p = round(grid_p - t_table_p, 3) if pd.notna(grid_p) else None
            role = "training_calibration_anchor" if grid_part == "Training" else ("unseen_validation_checkpoint" if grid_part == "Validation" else "independent_test_evaluation_station")
            station_comparisons.append({
                "test_no": t_no,
                "station_role": role,
                "station_md": t_md,
                "reconciled_station_p_bar": t_table_p,
                "conflicting_table_p_bar": None,
                "grid_md": grid_md,
                "distance_m": md_dist,
                "grid_target_bar": round(float(grid_p), 3) if pd.notna(grid_p) else None,
                "target_residual_bar": diff_p,
                "target_source_test_no": int(grid_source_test) if grid_source_test is not None else None,
                "target_provenance": grid_prov,
                "station_provenance": "measured_anchor",
                "partition": grid_part,
                "remarks": t_rem,
                "status": "matched" if diff_p is not None and abs(diff_p) < 0.1 else "unresolved_conflict",
            })

    # Discontinuities and gradient audit
    discontinuities = [
        {
            "boundary": "Hugin to Sleipner (3,263.50 m MD)",
            "top_formation_p": round(float(targets_df.loc[targets_df["FORMATION"] == "Hugin", "PP_TARGET_BAR"].iloc[-1]), 3),
            "base_formation_p": round(float(targets_df.loc[targets_df["FORMATION"] == "Sleipner", "PP_TARGET_BAR"].iloc[0]), 3),
            "step_bar": round(float(targets_df.loc[targets_df["FORMATION"] == "Sleipner", "PP_TARGET_BAR"].iloc[0] - targets_df.loc[targets_df["FORMATION"] == "Hugin", "PP_TARGET_BAR"].iloc[-1]), 3),
            "geological_nature": "Baffled step increase across regional seal hardground into Sleipner compartment.",
        },
        {
            "boundary": "Sleipner to Transition Zone (3,326.00 m MD)",
            "sleipner_base_p": round(float(targets_df.loc[targets_df["FORMATION"] == "Sleipner", "PP_TARGET_BAR"].iloc[-1]), 3),
            "transition_zone_policy": "Excluded from primary targets (IS_TARGET_VALID = False).",
            "geological_nature": "Basal hardground baffle followed by interbedded transition zone.",
        },
        {
            "boundary": "Transition Zone to Skagerrak (3,386.50 m MD)",
            "transition_zone_policy": "Excluded from primary targets (IS_TARGET_VALID = False).",
            "skagerrak_top_p": round(float(targets_df.loc[targets_df["FORMATION"] == "Skagerrak", "PP_TARGET_BAR"].iloc[0]), 3),
            "geological_nature": "Deep overpressure onset in Triassic Skagerrak reservoir sand.",
        },
    ]

    return {
        "total_observations": total_obs,
        "valid_target_count": valid_obs,
        "invalid_target_count": invalid_obs,
        "provenance_counts": prov_counts,
        "partition_summary": part_summary,
        "formation_summary": formation_summary,
        "missing_target_summary": missing_summary,
        "station_comparisons": station_comparisons,
        "discontinuities": discontinuities,
    }

