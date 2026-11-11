# Formation Pore-Pressure Target Generation Audit Report

This document presents the complete pre-training audit of the continuous pore-pressure target vectors generated for Well 15/9-F-12.

Date: 2026-11-11
Repository: wisdomdivine/final-year-project
Status: Target Vector Generation Completed; Pre-Training Review Deliverable

---

## 1. Executive Summary and Dataset Overview

- **Total Depth Observations:** 1811
- **Valid Target Observations (`IS_TARGET_VALID == True`):** 1414 (78.08%)
- **Invalid / Unsupported Observations (`IS_TARGET_VALID == False`):** 397 (21.92%)
- **Depth Coverage:** 3,127.7052 m to 3,403.5492 m MD (step = 0.1524 m)
- **Vertical TVD Coverage:** 2,819.38 m to 2,983.34 m TVD MSL (2,874.28 m to 3,038.24 m TVD RKB)

### Target Provenance Classification

| Provenance Category | Row Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `physics_derived` | 1287 | 71.07% | Continuous hydrostatic fluid columns (Hugin oil gradient; Sleipner compartment brine gradients) |
| `invalid_unsupported` | 397 | 21.92% | Unmeasured interbedded transition zone (3,326.0 to 3,386.5 m MD) excluded from primary targets |
| `provisional_assumption` | 110 | 6.07% | Skagerrak constant compartment assumption (368.762 bar); NOT a proven physical profile |
| `measured_anchor` | 17 | 0.94% | Exact grid points matching verified physical pretest measurements (12 Hugin, 3 Sleipner, 2 Skagerrak) |

---

## 2. Partition Breakdown and Anti-Leakage Audit

The target dataset adheres strictly to the contiguous 70:15:15 depth split. Zero station measurements from the Test partition were utilized in calibration, parameter fitting, or scaling.

| Partition | Total Rows | Valid Targets | Invalid Targets | MD Range (m) | Calibrating Anchors Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | 1268 | 1268 | 0 | 3127.71 - 3320.80 | Tests 1 to 12 (Hugin), Tests 14 & 15 (Sleipner) |
| **Validation** | 272 | 34 | 238 | 3320.95 - 3362.25 | Test 16 (held out as intermediate validation station) |
| **Testing** | 271 | 112 | 159 | 3362.40 - 3403.55 | Tests 20 & 21 (held out strictly for final physical evaluation) |

> [!IMPORTANT]
> **Strict Anti-Leakage Guarantee:** Tests 20 (368.706 bar) and 21 (368.818 bar) reside entirely within the Testing partition (rows 1540 to 1810). They have zero influence on training loss, feature transformations, standard scaling, hyperparameter tuning, or validation early stopping.

---

## 3. Formation Coverage and Pressure Statistics

| Formation | Total Rows | Valid Rows | MD Range (m) | TVD MSL Range (m) | Pressure Min (bar) | Pressure Max (bar) | Pressure Mean (bar) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hugin** | 892 | 892 | 3127.71 - 3263.49 | 2819.38 - 2899.64 | 320.020 | 325.702 | 322.835 |
| **Sleipner** | 410 | 410 | 3263.65 - 3325.98 | 2899.73 - 2936.74 | 332.283 | 341.796 | 335.379 |
| **Transition_Zone** | 397 | 0 | 3326.13 - 3386.48 | 2936.83 - 2973.09 | N/A (Excluded) | N/A (Excluded) | N/A (Excluded) |
| **Skagerrak** | 112 | 112 | 3386.63 - 3403.55 | 2973.18 - 2983.34 | 368.706 | 368.818 | 368.762 |

---

## 4. Sleipner-Skagerrak Transition Zone Analysis (MD 3,326.0 to 3,386.5 m)

- **Interval Span:** 3,326.1312 m to 3,386.4852 m MD (397 rows, 60.5 m length).
- **Target Status:** Strictly excluded from all primary targets (`IS_TARGET_VALID = False`; `PP_TARGET_BAR = NaN`).
- **Lithological Characterization:**
  - Silty shales, claystones, and mudstones ($V_{CL} \ge 0.50$): 302 rows (76.07%).
  - Silty sandstones and siltstone stringers ($V_{CL} < 0.50$): 95 rows (23.93%), with effective porosity up to 20.75% and minimum $V_{CL} = 0.197$.
  - **Correction Note:** This interval is an interbedded succession of silty shales, claystones, and thin silty sandstones, NOT a uniform mudstone package.
- **Physical Data Status:** Completely unsupported by permeable formation pressure tests. The only station in this interval is Test 17 at 3,333.57 m MD, which was a dry test (69.279 bar) confirming tight rock with no reservoir pressure.
- **Quarantined Sensitivity Scenarios:**
  - `PP_SENSITIVITY_RAMP_BAR`: Linear pressure ramp from 341.713 bar to 368.706 bar.
  - `PP_SENSITIVITY_TOPSEAL_BAR`: Sharp seal at the lithological Skagerrak boundary (3,340.0 m MD).
  - `PP_SENSITIVITY_BASALSEAL_BAR`: Sharp seal at the basal Sleipner hardground (3,326.3 m MD).
  - All sensitivity targets are stored in separate columns and excluded from primary model loss, validation early stopping, and test scoring.

---

## 5. Station-by-Station Target Verification

Comparison of generated continuous targets against all Schlumberger Run 10 pressure test stations, cleanly separating station-level role and provenance from grid-node target provenance:

| Test No | Station Role | Station MD (m) | Reconciled P (bar) | Table P (bar) | Grid MD (m) | Target P (bar) | Residual (bar) | Target Sourced From | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Test 1 | `training_calibration_anchor` | 3127.71 | 320.020 | - | 3127.71 | 320.020 | +0.000 | Test 1 | matched |
| Test 2 | `training_calibration_anchor` | 3127.99 | 320.042 | - | 3128.01 | 320.042 | +0.000 | Test 2 | matched |
| Test 3 | `training_calibration_anchor` | 3136.74 | 320.385 | - | 3136.70 | 320.385 | +0.000 | Test 3 | matched |
| Test 4 | `training_calibration_anchor` | 3144.07 | 320.713 | - | 3144.01 | 320.713 | +0.000 | Test 4 | matched |
| Test 5 | `training_calibration_anchor` | 3152.61 | 321.050 | - | 3152.55 | 321.050 | +0.000 | Test 5 | matched |
| Test 6 | `training_calibration_anchor` | 3170.73 | 321.827 | - | 3170.68 | 321.827 | +0.000 | Test 6 | matched |
| Test 7 | `training_calibration_anchor` | 3191.85 | 322.677 | - | 3191.87 | 322.677 | +0.000 | Test 7 | matched |
| Test 8 | `training_calibration_anchor` | 3221.68 | 323.892 | - | 3221.74 | 323.892 | +0.000 | Test 8 | matched |
| Test 9 | `training_calibration_anchor` | 3240.73 | 324.698 | - | 3240.79 | 324.698 | +0.000 | Test 9 | matched |
| Test 10 | `training_calibration_anchor` | 3254.85 | 325.298 | - | 3254.81 | 325.298 | +0.000 | Test 10 | matched |
| Test 11 | `training_calibration_anchor` | 3263.51 | 325.702 | - | 3263.49 | 325.702 | +0.000 | Test 11 | matched |
| Test 12 | `training_calibration_anchor` | 3262.78 | 325.686 | - | 3262.73 | 325.686 | +0.000 | Test 12 | matched |
| Test 13 | `aborted_pretest_reconciled` | 3293.55 | 334.161 | 328.056 | 3293.52 | 334.074 | -0.087 | Upper Sleipner Gradient | reconciled_via_plot |
| Test 14 | `training_calibration_anchor` | 3294.00 | 334.101 | - | 3293.97 | 334.101 | +0.000 | Test 14 | matched |
| Test 15 | `training_calibration_anchor` | 3307.73 | 335.610 | - | 3307.69 | 335.610 | +0.000 | Test 15 | matched |
| Test 16 | `unseen_validation_checkpoint` | 3324.65 | 341.713 | - | 3324.61 | 341.713 | +0.000 | Test 16 | matched |
| Test 17 | `dry_test_excluded` | 3333.57 | 69.279 | - | 3333.60 | NaN | N/A | None | unsupported_transition |
| Test 18 | `initial_pretest_fair_quality` | 3386.51 | 354.001 | - | 3386.48 | NaN | N/A | None | unsupported_transition |
| Test 19 | `alternative_memory_observation` | 3403.53 | 357.714 | - | 3403.55 | 368.818 | +11.104 | Test 21 | unresolved_memory_conflict |
| Test 20 | `independent_test_evaluation_station` | 3386.99 | 368.706 | - | 3386.94 | 368.706 | +0.000 | Test 20 | matched |
| Test 21 | `independent_test_evaluation_station` | 3403.57 | 368.818 | - | 3403.55 | 368.818 | +0.000 | Test 21 | matched |

### Key Station Observations
1. **Tests 1 to 12 (Hugin):** Perfect agreement at discrete anchor points (residual = 0.000 bar). Max residual of continuous regression across all 12 stations is 0.057 bar.
2. **Test 13 (Sleipner):** Original FPWD pressure-versus-time plot (p. 134) stabilized at 334.161 bar final buildup. Alternate table listing of 328.056 bar was recorded during intermediate drawdown investigation. Generated target (334.074 bar) yields a residual of -0.087 bar against the plot buildup, agreeing within 0.060 bar of companion repeat Test 14 (334.101 bar).
3. **Tests 14, 15, 16 (Sleipner):** Exact agreement at anchor stations (residual = 0.000 bar). Baffled compartments reflect measured step increases across calcite hardgrounds.
4. **Test 17 (3,333.57 m MD):** Correctly mapped as `dry_test_excluded` in the unsupported transition zone.
5. **Test 18 (3,386.51 m MD):** Initial incomplete pretest (354.001 bar) at transition boundary; companion repeat Test 20 stabilized at 368.706 bar.
6. **Test 19 vs Test 21 Decoupling (3,403.55 m MD):** Grid node 1810 target (368.818 bar) is strictly and exclusively sourced from Test 21 (`measured_anchor`). Test 19 (357.714 bar) is preserved as an uncalibrated memory recovery observation following real-time telemetry shutdown, and is strictly excluded from calibration anchors.
7. **Test 20 (3,386.99 m MD):** Primary measured anchor matching 368.706 bar exactly (residual = 0.000 bar) in Testing partition.
8. **Test 21 (3,403.57 m MD):** Primary measured anchor matching 368.818 bar exactly (residual = 0.000 bar) in Testing partition.

---

## 6. Discontinuities and Pressure Gradients

### Hugin to Sleipner (3,263.50 m MD)
- **Top formation p:** 325.702
- **Base formation p:** 332.283
- **Step bar:** 6.581
- **Geological nature:** Baffled step increase across regional seal hardground into Sleipner compartment.

### Sleipner to Transition Zone (3,326.00 m MD)
- **Sleipner base p:** 341.796
- **Transition zone policy:** Excluded from primary targets (IS_TARGET_VALID = False).
- **Geological nature:** Basal hardground baffle followed by interbedded transition zone.

### Transition Zone to Skagerrak (3,386.50 m MD)
- **Transition zone policy:** Excluded from primary targets (IS_TARGET_VALID = False).
- **Skagerrak top p:** 368.762
- **Geological nature:** Deep overpressure onset in Triassic Skagerrak reservoir sand.

---

## 7. Compliance Checklist Prior to Model Training

- [x] Explicit provenance metadata attached to all 1,811 observations.
- [x] Provisional Skagerrak targets distinguished from measured physical anchors.
- [x] Transition interval (3,326.0 to 3,386.5 m MD) excluded from all primary targets.
- [x] Documentation accurately describes transition interval as interbedded succession, not pure mudstone.
- [x] Tests 20 and 21 preserved as separate measured anchors.
- [x] Strict 70:15:15 partition isolation enforced; zero Test leakage into Training or Validation.
- [x] Target vectors and split files dual-saved in CSV and JSON formats.
