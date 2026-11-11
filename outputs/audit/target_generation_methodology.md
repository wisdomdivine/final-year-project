# Target-Generation Methodology and Geomechanical Calibration

This document establishes the technical, mathematical, and petrophysical specification for constructing formation pore-pressure target vectors in Well 15/9-F-12.

Date: 2026-11-11
Repository: wisdomdivine/final-year-project
Status: Corrected Technical Specification Prior to Vector Generation

---

## 1. Physical Regime Classification and Interval Formulations

The 1,811-observation study interval (3,127.7052 m to 3,403.5492 m MD) traverses three distinct geological formations with contrasting fluid types, compaction states, and pressure regimes.

### 1.1 Hugin Formation (3,127.7052 m to 3,263.5000 m MD)
- Lithology: Clean, porous, highly permeable reservoir sandstone (SAND_FLAG = 1.0, mean porosity 0.22 v/v, mean permeability 512 mD).
- Direct Physical Measurements: 12 Schlumberger Run 10 MWD pretest stations (Tests 1 to 12).
- Fluid Regime: Continuous, connected hydrostatic hydrocarbon (oil) column.
- Mathematical Formulation and Coordinate Reference Reconciliation:
  The linear regression across Tests 1 to 12 was fitted using the official Schlumberger pretest table (`Test_point_table.xls`), where the depth column `tvd_m` is referenced to the Rotary Kelly Bushing (RKB):
  
  1. Referenced to TVD RKB (RKB datum, +54.90 m MSL):
     P(TVD_RKB) = 118.2048 + 0.070276 * TVD_RKB (bar)
     At Test 1 (TVD_RKB = 2,871.76 m):
     P = 118.2048 + 0.070276 * 2871.76 = 320.020 bar (exact match to measured pretest).

  2. Referenced to TVD MSL (Mean Sea Level datum, TVD_MSL = TVD_RKB - 54.90 m):
     Substituting TVD_RKB = TVD_MSL + 54.90 m:
     P(TVD_MSL) = 118.2048 + 0.070276 * (TVD_MSL + 54.90)
     P(TVD_MSL) = (118.2048 + 0.070276 * 54.90) + 0.070276 * TVD_MSL
     P(TVD_MSL) = 122.0629 + 0.070276 * TVD_MSL (bar)
     At Test 1 (TVD_MSL = 2,816.86 m):
     P = 122.0629 + 0.070276 * 2816.86 = 320.020 bar (exact match to measured pretest).

  3. Discrepancy Explanation:
     Evaluating the RKB-referenced equation P = 118.205 + 0.07028 * z with a TVD_MSL value of 2,816.9 m omits the vertical elevation column between RKB and MSL (54.90 m * 0.070276 bar/m = 3.858 bar), yielding approximately 316.2 bar. The 3.86 bar offset is entirely attributable to this coordinate frame mismatch. The true MSL-referenced intercept is 122.063 bar.

- Fluid Gradient Validation:
  - Fitted slope: 0.070276 bar/m
  - Equivalent fluid density: rho_oil = 0.070276 / 0.0980665 = 0.7167 g/cm3
  - Equivalent API gravity: 42.6 deg API (central North Sea light crude)
  - Coefficient of determination: R2 = 0.999915
  - Max residual across all 12 test stations: 0.038 bar

### 1.2 Sleipner Formation (3,263.5000 m to 3,340.0000 m MD)
- Lithology: Interbedded fluvial sandstones, carbonaceous shales, and thin calcite-cemented stringers.
- Direct Physical Measurements:
  - Test 14 (3,294.00 m MD / 2,915.50 m TVD MSL): 334.101 bar (stabilized companion repeat for aborted Test 13).
  - Test 15 (3,307.73 m MD / 2,923.70 m TVD MSL): 335.610 bar.
  - Test 16 (3,324.65 m MD / 3,025.29 m TVD RKB / 2,970.39 m TVD MSL): 341.713 bar.
  - Test 17 (3,333.57 m MD): 69.279 bar (dry test in tight mudstone; excluded from calibration).
- Fluid Regime: Baffled, compartmentalized progression.
  - Sand intervals are segmented by continuous carbonate hardgrounds (CARB_FLAG = 1.0, density 2.50 to 2.66 g/cm3, DT 64 to 75 us/ft) identified at:
    - 3,286.5 to 3,290.8 m MD
    - 3,296.8 to 3,299.2 m MD
    - 3,326.3 to 3,330.1 m MD
  - These low-permeability hardgrounds act as vertical pressure baffles, maintaining discrete step increases in pressure across the Sleipner sand bodies.
- Mathematical Formulation:
  - Piecewise compartment anchoring: Sandstone lenses are anchored to their verified pretest measurements (334.10 bar, 335.61 bar, 341.71 bar).
  - Internal fluid gradients within individual sand bodies: 0.0981 to 0.1010 bar/m (formation brine).

### 1.3 Skagerrak Formation (3,340.0000 m to 3,403.5492 m MD)
- Lithology: Deep, overpressured Triassic sandstones and silty shales.
- Direct Physical Measurements:
  - Test 18 (3,386.51 m MD / 3,025.98 m TVD RKB): 354.001 bar (ASC Quality 0: initial incomplete pretest).
  - Test 20 (3,386.99 m MD / 3,026.27 m TVD RKB): 368.706 bar (ASC Quality 2 / report Good Test: stabilized companion repeat).
  - Test 19 (3,403.53 m MD / 3,036.23 m TVD RKB): 357.714 bar (ASC Quality 0 due to real-time telemetry shutdown; recovered from memory gauge as Good Test). Preserved as official memory reference.
  - Test 21 (3,403.57 m MD / 3,036.26 m TVD RKB): 368.818 bar (ASC Quality 2 / report Good Test: stabilized companion repeat). Primary physical reference.
- Skagerrak Pressure Gradient Inconsistency Analysis:
  - Vertical TVD separation between Test 20 and Test 21:
    Delta TVD = 3,036.26 - 3,026.27 = 9.99 m (under Trajectory 1: 2,983.34 - 2,973.36 = 9.98 m).
  - Pressure difference between Test 20 and Test 21:
    Delta P = 368.818 - 368.706 = 0.112 bar (11.2 kPa / 1.62 psi).
  - Apparent pressure gradient:
    g_apparent = 0.112 bar / 9.99 m = 0.01121 bar/m.
    Equivalent apparent fluid density:
    rho_apparent = 0.01121 / 0.0980665 = 0.1143 g/cm3.
  - Rejection of Formation-Brine Gradient (0.101 bar/m):
    A standard formation brine gradient of 0.101 bar/m (rho = 1.030 g/cm3) over 9.99 m of vertical depth separation requires a hydrostatic pressure increase of:
    Delta P_expected = 9.99 m * 0.1010 bar/m = 1.009 bar.
    The measured pressure difference of 0.112 bar is nearly an order of magnitude smaller (11% of the expected brine hydrostatic column).
    If a 0.101 bar/m brine gradient is anchored to Test 21 (368.818 bar), the predicted pressure at Test 20 would be 367.809 bar, which is 0.897 bar (13.0 psi) below the measured 368.706 bar.
    Conversely, if anchored to Test 20, Test 21 would be predicted at 369.715 bar (+0.897 bar above measured).
    Therefore, forcing a 0.101 bar/m brine gradient across Tests 20 and 21 introduces an artificial ~0.90 bar distortion.
  - Physical Interpretation of the Skagerrak Pressure Data:
    1. Instrumental Gauge Repeatability Limit: The Schlumberger FPWD quartz gauge has an operational repeatability margin of approximately +/- 0.05 to +/- 0.10 bar (+/- 0.7 to +/- 1.5 psi). Across a vertical interval of only 9.99 m, the expected fluid column delta is on the order of the tool repeatability limit.
    2. Approximately Constant Compartment Pressure Model: The physical data supports an approximately constant compartment pressure regime across the penetrated reservoir sand (mean P = 368.762 bar, with residuals of +/- 0.056 bar, entirely within gauge repeatability tolerance).
    3. Unresolved Fluid Model: While the apparent gradient of 0.0112 bar/m matches a light hydrocarbon gas column, petrophysical logs indicate a water-bearing/shaly reservoir. We conclude that the fluid gradient cannot be uniquely determined from two closely spaced stations separated by only 9.99 m TVD. We therefore reject the 0.101 bar/m brine model and model the interval as an approximately constant compartment pressure (368.762 bar) with an unresolved fluid gradient.
- Test 19 Operational Event and Memory Preservation:
  - During acquisition at 3,403.53 m MD, tool telemetry failed ("Unit shutdown"), resulting in zero surface data in real-time logs (`FM_PRESS_COMPUTED_MWD_1.ASC`, quality code 0).
  - Companion repeat Test 21 was immediately drilled 4 cm deeper (3,403.57 m MD), stabilized at 368.818 bar, and transmitted real-time data successfully.
  - Post-run downhole memory extraction recovered a complete drawdown and buildup record for Test 19: stabilized buildup pressure of 357.714 bar with drawdown mobility 5.37 mD/cP (volume 6.02 cc), classified as "Good Test" in Schlumberger's final summary table (`Test_point_table.xls`).
  - Test 19 is preserved as an official alternative memory reference observation. It conflicts with Test 21 by 11.104 bar. This discrepancy is preserved and documented without artificial discarding or forced reconciliation.
- Overpressure Recalculation under Corrected Trajectory 1:
  - Under corrected Trajectory 1, TVD MSL at the base of the predictor (MD 3,403.55 m / Test 21) is 2,983.34 m.
  - Normal formation brine hydrostatic pressure (density 1.030 g/cm3, seabed at 91.0 m MSL):
    Ph = 9.147 + 1.030 * 0.0980665 * (2983.34 - 91.0) = 301.30 bar.
  - Corrected Skagerrak Overpressure:
    Delta P = 368.818 - 301.30 = +67.52 bar (+979.3 psi).
  - (At Test 20: TVD MSL = 2,973.36 m, Ph = 300.29 bar; Delta P = 368.706 - 300.29 = +68.42 bar / +992.3 psi).
  - The previously cited figure of +63.76 bar was computed using the uncorrected Trajectory 2 (TVD MSL = 3,017.83 m, Ph = 304.78 bar) and is deprecated.

---

## 2. Target Provenance: Measured versus Constructed Values

The target dataset must rigorously document provenance across every interval to prevent synthetic labels from masquerading as physical ground truth:

| Interval (MD Range) | Anchor Measurements | Construction Method | Physical Assumptions | Uncertainty | Unsupported Regions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hugin Sandstone (3,127.71 to 3,263.50 m) | Tests 1 to 12 (320.02 to 325.70 bar) | Linear regression on TVD MSL: P = 122.063 + 0.07028 * TVD_MSL | Continuous hydrocarbon column; uniform oil density (0.7167 g/cm3) | +/- 0.05 bar in clean sand | Intercalated thin mudstones / coals (SAND_FLAG = 0) lack direct pressure; assumed in pressure equilibrium with bounding sand. |
| Sleipner Upper Sand (3,263.50 to 3,300.0 m) | Test 14 (334.10 bar at 3,294.0 m MD) | Anchor to Test 14 with brine gradient (0.101 bar/m) | Hydraulic isolation from Hugin by regional seal; internal sand continuity | +/- 0.50 bar | Upper boundary seal transition (3,263.5 to 3,293.5 m MD); unmeasured mudstones. |
| Sleipner Middle Sand (3,300.0 to 3,315.0 m) | Test 15 (335.61 bar at 3,307.7 m MD) | Anchor to Test 15 with brine gradient | Compartment bounded by hardground at 3,297 m | +/- 0.50 bar | Bounding hardground transitions. |
| Sleipner Lower Sand (3,315.0 to 3,326.0 m) | Test 16 (341.71 bar at 3,324.7 m MD) | Anchor to Test 16 with brine gradient | Isolated step compartment bounded by hardground at 3,326 m | +/- 0.80 bar | Lower Sleipner mudstones (3,326 to 3,340 m MD). |
| Sleipner-Skagerrak Transition (3,326.0 to 3,386.5 m) | None (Test 17 at 3,333.6 m is dry test, 69.28 bar) | Excluded from primary training target (IS_TARGET_VALID = False); secondary sensitivity experiment | Seal / compaction disequilibrium zone across 60.5 m unmeasured mudstones | High (+/- 10 to 15 bar if interpolated) | Completely unsupported by permeable pressure tests. Excluded from primary ML loss optimization. |
| Skagerrak Sand (3,386.5 to 3,403.55 m) | Tests 20 (368.71 bar), 21 (368.82 bar), and Test 19 memory (357.71 bar) | Constant compartment pressure model: P = 368.762 bar (+/- 0.06 bar) | Isolated overpressured Triassic reservoir compartment; unresolved fluid gradient | +/- 0.10 bar (gauge repeatability) | Top Skagerrak mudstones (3,340 to 3,386.5 m MD) lack direct tests. |

---

## 3. Evaluation of the Sleipner-Skagerrak Transition Zone (3,326.0 to 3,386.5 m MD)

The transition between the lower Sleipner sandstone (Test 16, 341.71 bar at 3,324.65 m MD) and the upper Skagerrak reservoir sand (Test 20, 368.71 bar at 3,386.99 m MD) represents a 60.5-meter interval (396 depth points at 0.1524 m spacing) characterized by a +27.0 bar pressure jump. The only test station within this interval is Test 17 at 3,333.57 m MD, which was a dry test (69.28 bar) confirming extremely tight formation permeability rather than reservoir pore pressure.

We evaluated three potential treatments for this unsupported interval:

### 3.1 Option 1: Exclude from Primary Training-Target Dataset
- Mechanism: Mask the target curve for the 3,326.0 to 3,386.5 m MD interval (`IS_TARGET_VALID = False` / NaN in primary target column).
- Advantages:
  - Eliminates target leakage and mathematical hallucinations: ML loss functions optimize strictly on physical ground-truth calibration data.
  - Protects validation integrity: In the spatial train/val/test split (Train: <= 3,320.8 m; Val: 3,320.95 to 3,362.25 m; Test: 3,362.40 to 3,403.55 m), this interval spans both the validation set (238 points) and upper test set (158 points). Evaluating model accuracy against synthetic interpolations rewards models that fit arbitrary mathematical functions rather than real physical laws.
- Disadvantages:
  - Results in an incomplete continuous target curve across the 60.5 m gap.

### 3.2 Option 2: Represent with an Explicitly Uncertain Interpolated Target
- Mechanism: Construct a continuous mathematical ramp (linear, cubic spline, or sigmoidal tanh function) connecting 341.71 bar to 368.71 bar, accompanied by an uncertainty band (+/- 8 to 12 bar).
- Advantages:
  - Provides a contiguous 1,811-point target vector.
- Disadvantages:
  - Severe risk of learning spurious correlations: Deep neural networks and gradient boosting machines minimize point-wise residuals. The model will force its weights to fit whatever shape is chosen (linear ramp or sigmoid), learning a synthetic relationship between petrophysical log features (GR, RHOB, DT) and pressure that has no empirical validation.

### 3.3 Option 3: Treat Using a Separate Sensitivity Experiment
- Mechanism: Define distinct physical transition scenarios:
  - Scenario A (Basal Sleipner Seal): Sharp step at the basal hardground (3,326.3 m MD).
  - Scenario B (Top Skagerrak Seal): Sharp step at the lithological Skagerrak boundary (3,340.0 m MD).
  - Scenario C (Gradual Compaction Disequilibrium): Distributed linear ramp across the full mudstone package.
- Advantages:
  - Quantifies model prediction variance and out-of-distribution stability under alternative geological configurations without polluting benchmark evaluation.

### 3.4 Formal Recommendation and Justification
We formally recommend **Option 1 (Exclusion from primary training-target dataset) paired with Option 3 (treatment as a secondary sensitivity diagnostic)**.

**Justification:**
1. Methodological Rigor: In high-stakes geomechanical pore-pressure prediction, presenting an interpolated line across an unmeasured 60.5 m sealing interval as a training ground truth is scientifically flawed.
2. Protection of Split Evaluation: Because the transition zone sits across the validation and test partitions, excluding it from primary training loss and validation loss ensures that model selection and testing are evaluated solely against verified physical pretests (Tests 16, 20, 21).
3. Best of Both Worlds: The primary pore-pressure target vector `PP_TARGET_BAR` will mask out this interval, while secondary diagnostic columns (`PP_SENSITIVITY_RAMP_BAR` and `PP_SENSITIVITY_TOPSEAL_BAR`) will be preserved in the dataset for diagnostic sensitivity analyses.

---

## 4. Independent Validation and Calibration Split

To enforce zero target leakage and prevent circular validation, pressure stations are divided strictly into training calibration and independent evaluation subsets:

### 4.1 Training Calibration Set (Tests 1 to 15, MD <= 3,320.80 m)
- Tests 1 to 12 (Hugin): Used to calibrate Hugin continuous target parameters.
- Tests 13 to 15 (Sleipner): Used to calibrate Sleipner upper/middle sand compartment baselines.
- Available for model parameter training.

### 4.2 Unseen Validation Set (Tests 16 and 17, 3,320.95 to 3,362.25 m MD)
- Test 16 (341.71 bar at 3,324.65 m MD): Held out from training as an unseen intermediate validation checkpoint.
- Test 17 (69.28 bar dry test): Excluded from quantitative calibration.

### 4.3 Completely Independent Evaluation Set (Tests 18 to 21, 3,362.40 to 3,403.55 m MD)
- Tests 20 (368.71 bar) and 21 (368.82 bar) are strictly reserved as independent ground-truth evaluation stations.
- Test 18 (354.00 bar) is preserved as an initial incomplete pretest (Fair Test).
- Test 19 (357.71 bar) is preserved as an official alternative memory reference observation, documenting the tool telemetry shutdown and memory recovery.
- Rule: No test partition station measurement may be used to train models, fit feature scalers, or tune Eaton exponents.
- Reporting Rule: ML performance on the test set MUST be evaluated against the discrete physical pretest measurements (Tests 20 and 21), rather than solely against generated continuous curves.

---

## 5. Formation Boundaries and Geological Assumptions: Observations versus Interpretations

### 5.1 Established Physical Observations
1. MWD pretest logs (`FM_PRESS_COMPUTED_MWD_1.ASC` and `Test_point_table.xls`) officially categorize:
   - Tests 1 to 12 as Hugin Formation.
   - Tests 13 to 17 as Sleipner Formation.
   - Tests 18 to 24 as Skagerrak Formation.
2. Petrophysical logs identify discrete carbonate-cemented stringers (`CARB_FLAG == 1.0` in `WLC_PETRO_COMPUTED_INPUT_1.LAS`) with high bulk density (2.50 to 2.66 g/cm3), low acoustic transit time (DT 64 to 75 us/ft), and near-zero porosity.
3. Stabilized formation pressure increases in steps:
   - Hugin oil column: 320.02 to 325.70 bar.
   - Sleipner sands: 334.10 bar, 335.61 bar, 341.71 bar.
   - Skagerrak reservoir: 368.71 to 368.82 bar.

### 5.2 Geological Interpretations
1. Calcite Hardground Baffles:
   - Interpreting the log-detected carbonate spikes as laterally extensive, sealing hardgrounds that isolate vertical pressure compartments is an interpretation. While supported by the measured step-pressures, lateral continuity cannot be established from a single 1D borehole without 3D seismic or multi-well interference testing.
2. Formation Boundary Depths:
   - Placing the Hugin/Sleipner boundary at 3,263.50 m MD and Sleipner/Skagerrak boundary at 3,340.00 m MD is an operational interpretation based on log facies transitions.
3. Transition Zone Gradients:
   - Assuming a smooth or sigmoidal pressure ramp across the unmeasured mudstones between 3,326 m and 3,386 m MD is an unverified model assumption.
