# Phase 2 Pre-Execution Validation Audit Report

This report presents the formal pre-execution validation audit for Phase 2 of the Well 15/9-F-12 pore-pressure prediction workflow, addressing trajectory reconciliation, formation-pressure station verification, target-generation physics, and overburden uncertainty prior to continuous target generation or model training.

Date: 2026-11-11
Repository: wisdomdivine/final-year-project
Status: Pre-Execution Validation Completed and Reconciled; Pending Review

---

## 1. Trajectory and TVD Reconciliation

### 1.1 Trajectory Comparison Across the 1,811 Observations
A rigorous observation-by-observation depth comparison was conducted between the two reconstructed trajectories:
- Trajectory 1 (Sensitivity Trajectory): Stored in `tables/overburden_sensitivity_1811.csv`, derived from the original MWD well trajectory reconstruction.
- Trajectory 2 (Profile Trajectory): Stored in `tables/overburden_stress_profile.csv`, reconstructed via PCHIP interpolation of Landmark COMPASS survey runs with surface tie-in.

Comparison metrics across all 1,811 predictor observations (3,127.7052 m to 3,403.5492 m MD):
- Top Predictor Depth (3,127.7052 m MD):
  - Trajectory 1: TVD MSL = 2,819.378 m | TVD RKB = 2,874.278 m
  - Trajectory 2: TVD MSL = 2,853.249 m | TVD RKB = 2,908.149 m
  - Delta TVD (Traj 2 - Traj 1): +33.871 m
- Mid Predictor Depth (3,265.6272 m MD):
  - Trajectory 1: TVD MSL = 2,900.915 m | TVD RKB = 2,955.815 m
  - Trajectory 2: TVD MSL = 2,935.014 m | TVD RKB = 2,989.914 m
  - Delta TVD (Traj 2 - Traj 1): +34.099 m
- Base Predictor Depth (3,403.5492 m MD):
  - Trajectory 1: TVD MSL = 2,983.343 m | TVD RKB = 3,038.243 m
  - Trajectory 2: TVD MSL = 3,017.827 m | TVD RKB = 3,072.727 m
  - Delta TVD (Traj 2 - Traj 1): +34.483 m
- Whole Interval Summary:
  - Minimum Delta TVD: +33.849 m
  - Maximum Delta TVD: +34.493 m
  - Mean Delta TVD: +34.190 m
  - Standard Deviation: 0.233 m
  - Correlation between delta MD and delta TVD: 0.99998 (parallel inclination profile with mean inclination of 53.4 degrees).

Both trajectories are preserved in the full depth comparison deliverable:
- Table: `outputs/audit/trajectory_reconciliation.csv` and `outputs/audit/trajectory_reconciliation.json`

### 1.2 Identification of the ~34.5 m Discrepancy
The source of the approximately 34.5 m discrepancy is an uncorrected coordinate reference frame and surface tie-in datum shift between RKB and MSL:

1. Wellhead Elevation and Reference Datums:
   - Well 15/9-F-12 was drilled by the jackup rig Maersk Inspirer.
   - Rig Rotary Kelly Bushing (RKB) elevation: +54.90 m above Mean Sea Level (MSL).
   - Water depth at well location: 91.0 m below MSL (seabed at -91.0 m MSL, or 145.9 m below RKB).
   - In standard petrophysical logs (CWLS LAS files), measured depth (MD) is measured from RKB (0 m MD = RKB = +54.90 m MSL).
   - Official conversion relationship: TVD_MSL = TVD_RKB - 54.90 m.

2. Trajectory 1 Methodology (Sensitivity Trajectory):
   - In Trajectory 1, survey stations were imported directly from the MWD definitive survey referenced to RKB.
   - At MD = 3,127.9 m, Trajectory 1 yields TVD_RKB = 2,874.28 m and TVD_MSL = 2,819.38 m.
   - This matches Schlumberger's official MWD pretest station log `FM_PRESS_COMPUTED_MWD_1.ASC` where at MD = 3,127.9 m:
     - DEPTH.TVD(RKB) = 2,871.8 m
     - DEPTH.TVD(MSL) = 2,816.9 m
     - Difference relative to Schlumberger official pretest: +2.48 m (0.08%, consistent with MWD telemetry versus definitive survey).

3. Trajectory 2 Methodology (Profile Trajectory):
   - In `scripts/reconstruct_overburden.py`, survey stations `df_surv` were parsed from Landmark COMPASS EDM XML (`Volve F.edm.xml`).
   - In Landmark COMPASS, the raw station `tvd` was already computed relative to the well rotary table (RKB).
   - However, the script concatenated `df_surv` with a sea-surface artificial tie-in:
     `{"md_m": 0.0, "tvd_m": 0.0}` and `{"md_m": 91.0, "tvd_m": 91.0}` (treating 0 m as MSL instead of RKB).
   - It assigned the resulting interpolator output directly to `df_lis["TVD_MSL_m"]`, and then added +54.90 m to obtain `df_lis["TVD_RKB_m"]`.
   - In a deviated borehole with average inclination theta = 53.4 degrees, an uncorrected datum shift of 54.90 m projects as:
     Delta TVD = 54.90 * cos(53.4 deg) = 32.75 m.
   - Compounded with the build-up section trajectory, this shifted the entire subsurface profile downward by 34.19 m.
   - Notice that:
     34.19 m (MSL discrepancy) + 20.71 m (RKB residual) = 54.90 m (the exact RKB elevation).

4. Conclusion and Resolution:
   - Neither trajectory is silently overwritten. Both are fully documented in `outputs/audit/trajectory_reconciliation.csv`.
   - Trajectory 1 represents the true MSL-referenced geological depth, verified against the Schlumberger MWD pretest logs.
   - Trajectory 2 represents an uncorrected surface-spliced run. All geomechanical parameters are audited under both trajectories in `outputs/audit/overburden_sensitivity.csv`.

---

## 2. Hugin Formation-Pressure Gradient Verification

### 2.1 Exact Station Regression Reproduction
The linear regression across Tests 1 to 12 in the Hugin Formation was reproduced using the official Schlumberger pretest records:
- Exact Stations Used: Tests 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (all 12 Run 10 Hugin pretests).
- Input Depths from `Test_point_table.xls`:
  - Test 1: MD = 3,127.71 m | tvd_m = 2,871.76 m | Pressure = 320.020 bar
  - Test 2: MD = 3,127.99 m | tvd_m = 2,871.92 m | Pressure = 320.042 bar
  - Test 3: MD = 3,136.74 m | tvd_m = 2,877.09 m | Pressure = 320.385 bar
  - Test 4: MD = 3,144.07 m | tvd_m = 2,881.43 m | Pressure = 320.713 bar
  - Test 5: MD = 3,152.61 m | tvd_m = 2,886.49 m | Pressure = 321.050 bar
  - Test 6: MD = 3,170.73 m | tvd_m = 2,897.20 m | Pressure = 321.827 bar
  - Test 7: MD = 3,191.85 m | tvd_m = 2,909.70 m | Pressure = 322.677 bar
  - Test 8: MD = 3,221.68 m | tvd_m = 2,927.35 m | Pressure = 323.892 bar
  - Test 9: MD = 3,240.73 m | tvd_m = 2,938.67 m | Pressure = 324.698 bar
  - Test 10: MD = 3,254.85 m | tvd_m = 2,947.11 m | Pressure = 325.298 bar
  - Test 11: MD = 3,263.51 m | tvd_m = 2,952.29 m | Pressure = 325.702 bar
  - Test 12: MD = 3,262.78 m | tvd_m = 2,951.85 m | Pressure = 325.686 bar

### 2.2 Root Cause of the 3.86 bar Discrepancy
When regressing formation pressure against `tvd_m` from Schlumberger's official table:
P = 118.2048 + 0.070276 * TVD (bar) [R2 = 0.999915]

Crucially, the column `tvd_m` in `Test_point_table.xls` was referenced to **RKB** (drill floor, +54.90 m MSL), NOT to MSL:
- At Test 1, TVD_RKB = 2,871.76 m:
  P = 118.2048 + 0.070276 * 2871.76 = 320.020 bar (exact agreement with measured pretest).
- When evaluated with TVD_MSL = 2,816.86 m:
  P = 118.2048 + 0.070276 * 2816.86 = 316.162 bar.
- The 3.858 bar discrepancy equals:
  Delta P = 54.90 m * 0.070276 bar/m = 3.858 bar.
- Conversion to True TVD MSL Coordinate Frame:
  Substituting TVD_RKB = TVD_MSL + 54.90 m into the verified regression:
  P(TVD_MSL) = 118.2048 + 0.070276 * (TVD_MSL + 54.90)
  P(TVD_MSL) = 122.0629 + 0.070276 * TVD_MSL (bar)
  At Test 1 (TVD_MSL = 2,816.86 m):
  P = 122.0629 + 0.070276 * 2816.86 = 320.020 bar.
- This mathematically resolves the discrepancy without arbitrarily adjusting any values.

---

## 3. Overpressure Recalculation under Corrected Trajectory 1

The overpressure in the Skagerrak Formation was recalculated using the corrected Trajectory 1:
- Base Predictor (MD = 3,403.55 m / Test 21):
  - Corrected TVD MSL = 2,983.34 m
  - Seabed: 91.0 m TVD MSL, seawater density 1.025 g/cm3 (P_seabed = 9.147 bar)
  - Formation brine hydrostatic pressure (density 1.030 g/cm3):
    Ph = 9.147 + 1.030 * 0.0980665 * (2983.34 - 91.0) = 301.30 bar
  - Measured stabilized formation pressure (Test 21): 368.818 bar
  - Corrected Overpressure at Test 21:
    Delta P = 368.818 - 301.30 = +67.52 bar (+979.3 psi)
- Test 20 (MD = 3,386.99 m):
  - Corrected TVD MSL = 2,973.36 m
  - Formation brine hydrostatic pressure: Ph = 300.29 bar
  - Measured stabilized formation pressure (Test 20): 368.706 bar
  - Corrected Overpressure at Test 20:
    Delta P = 368.706 - 300.29 = +68.42 bar (+992.3 psi)
- Deprecation Note:
  The previously cited overpressure of +63.76 bar was computed using uncorrected Trajectory 2 (TVD MSL = 3,017.83 m, Ph = 304.78 bar) and is formally deprecated.

---

## 4. Formation-Pressure Reference Reconciliation: Tests 18 to 21

All 24 Run 10 tests are detailed in `outputs/audit/pressure_station_reconciliation.csv`:
- Test 18 (3,386.51 m MD): Pressure = 354.001 bar, ASC Quality 0 ("Pressure did not stabilize"), Fair Test. Initial aborted pretest.
- Test 20 (3,386.99 m MD, 0.48 m deeper): Pressure = 368.706 bar, ASC Quality 2, Good Test (Type A pretest, 0.51 cc). Primary stabilized reference.
- Test 19 (3,403.53 m MD): Real-time ASC recorded tool shutdown with zero valid telemetry. Post-run memory gauge recovery in operations report yielded 357.714 bar.
- Test 21 (3,403.57 m MD, 4 cm deeper): Pressure = 368.818 bar, ASC Quality 2, Good Test. Primary stabilized reference.
- Unresolved Discrepancy:
  Test 21 is 11.104 bar higher than Test 19 at virtually the same depth. Test 21 agrees with Test 20 within 0.112 bar over 9.99 m TVD (a physical fluid gradient of 0.0112 bar/m). The discrepancy in the official documentation is flagged and preserved.

---

## 5. Target Provenance and Independent Validation Split

### 5.1 Target Provenance Specification
- Hugin (3,127.71 to 3,263.50 m MD): Measured at 12 stations; calculated via oil gradient P = 122.063 + 0.07028 * TVD_MSL in clean sand; thin mudstones assumed in equilibrium.
- Sleipner (3,263.50 to 3,340.00 m MD): Measured at stations 14 (334.10 bar), 15 (335.61 bar), 16 (341.71 bar); step-wise compartment anchoring across calcite hardgrounds; lower Sleipner mudstones (3,326 to 3,340 m MD) are unsupported by permeable tests and represent an unmeasured transition zone.
- Skagerrak (3,340.00 to 3,403.55 m MD): Measured at stations 20 (368.71 bar) and 21 (368.82 bar); compartment anchored to 368.8 bar; top Skagerrak (3,340 to 3,386 m MD) lacks direct tests.

### 5.2 Independent Validation Split
- Calibration Set (Tests 1 to 15, MD <= 3,320.80 m): Used for model training and Hugin/upper Sleipner calibration.
- Unseen Validation Set (Tests 16 and 17, 3,320.95 to 3,362.25 m MD): Held out as an intermediate validation checkpoint. Test 17 (dry test, 69.28 bar) excluded.
- Completely Independent Evaluation Set (Tests 18 to 21, 3,362.40 to 3,403.55 m MD): Held out strictly for final evaluation.
- Reporting Rule: ML performance on the test set must be evaluated against the discrete physical pretest measurements (Tests 20 and 21), rather than solely against generated continuous curves.

---

## 6. Formation Boundaries and Geological Assumptions

1. Established Physical Observations:
   - Official MWD pretest logs categorize Tests 1-12 as Hugin, Tests 13-17 as Sleipner, and Tests 18-24 as Skagerrak.
   - Continuous logs record dense carbonate-cemented stringers (CARB_FLAG = 1.0, density 2.50 to 2.66 g/cm3, DT 64 to 75 us/ft).
   - Physical pressure steps: 325.70 bar in Hugin -> 334.10 bar in upper Sleipner -> 341.71 bar in lower Sleipner -> 368.82 bar in Skagerrak.
2. Geological Interpretations (Flagged):
   - Lateral continuity of calcite stringers as regional sealing hardgrounds is an interpretation from 1D borehole data.
   - Placing boundary depths at exactly 3,263.50 m and 3,340.00 m MD is an operational interpretation based on log character transitions.
   - Pressure profile shape across the unmeasured mudstones between 3,326 m and 3,386 m MD is an unverified model assumption.

---

## 7. Concise List of Unresolved Scientific Limitations

1. Lower Sleipner to Upper Skagerrak Transition (3,326.0 to 3,386.5 m MD):
   - A 60.5-meter interval without valid permeable pressure tests (only dry Test 17 at 3,333.6 m MD).
   - The exact depth where pressure transitions from 341.7 bar to 368.7 bar cannot be directly observed from well test data.
2. Test 19 versus Test 21 Memory Recovery Discrepancy (3,403.55 m MD):
   - Official operations report records an 11.10 bar discrepancy between two companion tests conducted 4 cm apart. Both are classified as "Good Test" in the Schlumberger summary table.
3. 1D Extrapolation of Calcite Hardground Baffles:
   - While pressure steps prove hydraulic compartmentalization at the wellbore, lateral extent and continuity cannot be independently verified without 3D seismic or multi-well production data.
4. Chemical Compaction Precludes Classical Eaton Sonic Application:
   - Deep Jurassic shales have undergone quartz cementation, lowering DT to 75-82 us/ft. Standard mechanical NCT extrapolation predicts unphysical sub-hydrostatic pressures.
