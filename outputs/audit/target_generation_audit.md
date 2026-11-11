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

Comparison of generated continuous targets against all Schlumberger Run 10 pressure test stations:

| Test No | Formation | Station MD (m) | Grid MD (m) | Dist (m) | Measured P (bar) | Generated Target (bar) | Residual (bar) | Provenance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Test 1 | - | 3127.71 | 3127.71 | 0.0048 | 320.020 | 320.020 | +0.000 | `measured_anchor` | matched |
| Test 2 | - | 3127.99 | 3128.01 | 0.0200 | 320.042 | 320.042 | +0.000 | `measured_anchor` | matched |
| Test 3 | - | 3136.74 | 3136.70 | 0.0432 | 320.385 | 320.385 | +0.000 | `measured_anchor` | matched |
| Test 4 | - | 3144.07 | 3144.01 | 0.0580 | 320.713 | 320.713 | +0.000 | `measured_anchor` | matched |
| Test 5 | - | 3152.61 | 3152.55 | 0.0636 | 321.050 | 321.050 | +0.000 | `measured_anchor` | matched |
| Test 6 | - | 3170.73 | 3170.68 | 0.0480 | 321.827 | 321.827 | +0.000 | `measured_anchor` | matched |
| Test 7 | - | 3191.85 | 3191.87 | 0.0156 | 322.677 | 322.677 | +0.000 | `measured_anchor` | matched |
| Test 8 | - | 3221.68 | 3221.74 | 0.0560 | 323.892 | 323.892 | +0.000 | `measured_anchor` | matched |
| Test 9 | - | 3240.73 | 3240.79 | 0.0560 | 324.698 | 324.698 | +0.000 | `measured_anchor` | matched |
| Test 10 | - | 3254.85 | 3254.81 | 0.0432 | 325.298 | 325.298 | +0.000 | `measured_anchor` | matched |
| Test 11 | - | 3263.51 | 3263.49 | 0.0164 | 325.702 | 325.702 | +0.000 | `measured_anchor` | matched |
| Test 12 | - | 3262.78 | 3262.73 | 0.0484 | 325.686 | 325.686 | +0.000 | `measured_anchor` | matched |
| Test 13 | - | 3293.55 | 3293.52 | 0.0336 | 328.056 | 334.074 | +6.018 | `physics_derived` | unresolved_conflict |
| Test 14 | - | 3294.00 | 3293.97 | 0.0264 | 334.101 | 334.101 | +0.000 | `measured_anchor` | matched |
| Test 15 | - | 3307.73 | 3307.69 | 0.0404 | 335.610 | 335.610 | +0.000 | `measured_anchor` | matched |
| Test 16 | - | 3324.65 | 3324.61 | 0.0440 | 341.713 | 341.713 | +0.000 | `measured_anchor` | matched |
| Test 17 | - | 3333.57 | 3333.60 | 0.0276 | 69.279 | NaN | N/A | `invalid_unsupported` | unsupported_transition |
| Test 18 | - | 3386.51 | 3386.48 | 0.0296 | 354.001 | NaN | N/A | `invalid_unsupported` | unsupported_transition |
| Test 19 | - | 3403.53 | 3403.55 | 0.0192 | 357.714 | 368.818 | +11.104 | `measured_anchor` | unresolved_conflict |
| Test 20 | - | 3386.99 | 3386.94 | 0.0524 | 368.706 | 368.706 | +0.000 | `measured_anchor` | matched |
| Test 21 | - | 3403.57 | 3403.55 | 0.0208 | 368.818 | 368.818 | +0.000 | `measured_anchor` | matched |

### Key Station Observations
1. **Tests 1 to 12 (Hugin):** Perfect agreement at discrete anchor points (residual = 0.000 bar). Max residual of the continuous regression across all 12 stations is 0.057 bar.
2. **Tests 14, 15, 16 (Sleipner):** Exact agreement at anchor stations (residual = 0.000 bar). Baffled compartments reflect measured step increases across calcite hardgrounds.
3. **Test 17 (3,333.57 m MD):** Correctly mapped as `invalid_unsupported` in the transition zone.
4. **Test 20 (3,386.99 m MD):** Measured anchor matching 368.706 bar exactly (residual = 0.000 bar).
5. **Test 21 (3,403.57 m MD):** Measured anchor matching 368.818 bar exactly (residual = 0.000 bar).
6. **Test 19 (3,403.53 m MD):** Preserved in station table (357.714 bar). Documented as an unresolved operational conflict (11.104 bar lower than Test 21). Strictly excluded from primary calibration.

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
