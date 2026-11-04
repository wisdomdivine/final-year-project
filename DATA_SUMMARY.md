# Dataset Summary and Experimental Design

## 1. Predictor Dataset Verification

The predictor dataset was reconstructed from the official Equinor Volve Field LAS files for well 15/9-F-12. The interval covers 3127.7052 m to 3403.5492 m at a sampling rate of 0.1524 m, yielding exactly 1,811 continuous rows with zero missing values.

Below is the comparison between the reconstructed predictor statistics and the published values from Siddique et al.:

| Parameter / Variable | Reconstructed Data | Siddique et al. Published | Match Status |
|:---------------------|:-------------------|:--------------------------|:-------------|
| Total Observations   | 1,811              | 1,811                     | Exact        |
| Depth Start (m)      | 3127.71            | 3127.71                   | Exact        |
| Depth End (m)        | 3403.55            | 3403.55                   | Exact        |
| Depth Mean (m)       | 3265.6272          | 3265.63                   | Exact        |
| DT Mean (us/ft)      | 82.9246            | 82.92                     | Exact        |
| GR Mean (API)        | 64.1874            | 64.19                     | Exact        |
| GR Range (API)       | 22.45 to 122.87    | 22.45 to 122.87           | Exact        |
| RD Mean (ohm.m)      | 52.9025            | 52.90                     | Exact        |
| RD Max (ohm.m)       | 1319.27            | 1319.27                   | Exact        |
| ROP Mean (m/hr)      | 17.9755            | 17.98                     | Exact        |
| RS Mean (ohm.m)      | 62.0130            | 62.01                     | Exact        |
| RT Mean (ohm.m)      | 52.6648            | 52.66                     | Exact        |
| K Mean (mD)          | 142.0355           | 142.04                    | Exact        |
| K Max (mD)           | 12109.04           | 12109.04                  | Exact        |
| SW Mean (v/v)        | 0.5023             | 0.50                      | Exact        |
| VCL Mean (v/v)       | 0.3960             | 0.40                      | Exact        |

## 2. Input Predictor Variables (X)

The reconstructed feature matrix X consists of 14 continuous variables:

| Variable | Description | Source Curve / Mapping | Units |
|:---------|:------------|:-----------------------|:------|
| Depth    | Measured depth | DEPTH | m |
| DT       | Compressional sonic slowness | DT | us/ft |
| GR       | Gamma ray | GR | API |
| NPHI     | Neutron porosity | NPHI | v/v |
| RD       | Deep resistivity | RD | ohm.m |
| RHOB     | Bulk density | RHOB | g/cm3 |
| ROP      | Rate of penetration | ROP5_RM mapped to ROP | m/hr |
| RS       | Shallow resistivity | RS | ohm.m |
| RT       | True resistivity | RT | ohm.m |
| BVW      | Bulk volume water | BVW | v/v |
| K        | Permeability | KLOGH mapped to K | mD |
| PHIF     | Formation porosity | PHIF | v/v |
| SW       | Water saturation | SW | v/v |
| VCL      | Clay volume | VSH mapped to VCL | v/v |

## 3. Contiguous Depth Partitioning

To mitigate spatial autocorrelation leakage, the 1,811 sequential observations are split chronologically down-hole:

| Dataset Partition | Percentage | Rows | Depth Interval (m) | Primary Purpose |
|:------------------|:-----------|:-----|:-------------------|:----------------|
| Training          | 70%        | 1,268 | 3127.71 to 3320.80 | Model parameter learning |
| Validation        | 15%        | 272  | 3320.95 to 3362.25 | Hyperparameter tuning and model selection |
| Testing           | 15%        | 271  | 3362.40 to 3403.55 | Final unbiased model evaluation |

## 4. Multicollinearity Assessment (Variance Inflation Factor)

Multicollinearity was evaluated across all 14 variables using the Variance Inflation Factor (VIF). Values greater than 10 indicate severe collinearity that can destabilize linear regression coefficients:

| Variable | VIF | Multicollinearity Level | Primary Correlation Drivers |
|:---------|:----|:------------------------|:----------------------------|
| PHIF     | 107.62 | Severe               | Strongly anti-correlated with RHOB (-0.982) |
| RHOB     | 71.81  | Severe               | Strongly anti-correlated with PHIF (-0.982) |
| GR       | 39.96  | Severe               | Strongly correlated with VCL (0.912) |
| SW       | 37.06  | Severe               | Correlated with RHOB (0.893) and PHIF (-0.870) |
| VCL      | 23.64  | Severe               | Strongly correlated with GR (0.912) |
| BVW      | 9.79   | Moderate             | Related to porosity and saturation |
| RT       | 7.74   | Moderate             | Correlated with RD (0.912) |
| DEPTH    | 7.30   | Moderate             | Depth trend across formation compaction |
| ROP      | 6.70   | Moderate             | Moderately correlated with SW (0.832) |
| NPHI     | 6.30   | Moderate             | Correlated with PHIF (0.809) |
| RD       | 6.15   | Moderate             | Correlated with RT (0.912) |
| DT       | 1.80   | Low                  | Weak linear dependency |
| RS       | 1.20   | Low                  | Flushed zone resistivity |
| K        | 1.20   | Low                  | Dominated by reservoir intervals |

Implications for modeling:
- Tree-based models (Random Forest, CatBoost) are naturally robust to multicollinearity because splits are determined on individual features.
- Multiple Linear Regression (MLR) coefficients will be unstable if all collinear features are included without regularization. An L2 regularized variant (Ridge) or feature selection may be assessed alongside ordinary MLR.

## 5. Partition Distribution Comparison

A key feature of the contiguous depth split is that the testing set evaluates model generalization to a deeper subsurface interval:

| Variable | Training Mean | Validation Mean | Testing Mean | Overall Mean | Observation on Distribution Shift |
|:---------|:--------------|:----------------|:-------------|:-------------|:-----------------------------------|
| DEPTH (m) | 3224.25     | 3341.60         | 3382.98      | 3265.63      | Contiguous depth progression down-hole |
| DT (us/ft) | 84.49       | 78.90           | 79.65        | 82.92        | Slightly faster formation in deeper intervals |
| GR (API)  | 56.83       | 83.71           | 79.01        | 64.19        | Shadier lithology in deeper intervals |
| NPHI (v/v) | 0.21       | 0.18            | 0.18         | 0.20         | Lower apparent porosity deeper |
| RD (ohm.m) | 74.94      | 1.60            | 1.27         | 52.90        | Hydrocarbon column shallow, brine deeper |
| RHOB (g/cm3) | 2.30     | 2.50            | 2.47         | 2.35         | Denser rock in deeper intervals |
| RS (ohm.m) | 87.82      | 2.04            | 1.45         | 62.01        | Matches deep resistivity pattern |
| RT (ohm.m) | 74.33      | 2.10            | 2.04         | 52.66        | Matches deep resistivity pattern |
| ROP (m/hr) | 14.71      | 26.08           | 25.12        | 17.98        | Higher penetration rate in deeper section |
| BVW (v/v) | 0.05        | 0.09            | 0.11         | 0.07         | Increased bulk water volume deeper |
| K (mD)    | 180.36      | 102.83          | 2.06         | 142.04       | Low permeability beneath reservoir base |
| PHIF (v/v) | 0.20       | 0.10            | 0.13         | 0.18         | Porosity drops below main reservoir |
| SW (v/v)  | 0.33        | 0.92            | 0.87         | 0.50         | Oil-water contact transition |
| VCL (v/v) | 0.36        | 0.52            | 0.45         | 0.40         | Increase in clay volume |

This distribution shift reflects genuine reservoir geology: the shallower 70% includes the oil-bearing Hugin sandstone, while the deeper 15% penetrates the underlying transition and water-bearing formation. This makes the depth-based test a rigorous real-world evaluation of model transferability.

## 6. Preprocessing and Transformation Strategy

1. Skewness handling:
   - High positive skewness in permeability K (11.66) and resistivity logs RS (11.96), RD (4.29), and RT (3.57) justifies evaluating log10 transformations for distance-sensitive and linear architectures (MLR, MLP).
2. Feature scaling:
   - Standardization parameters (mean and standard deviation) are strictly computed on the 70% Training set (1,268 rows) and applied without recomputation to Validation and Testing sets.
   - Tree ensembles (Random Forest, CatBoost) are invariant to monotonic scaling and will be evaluated on native scales.

## 7. Candidate Algorithms and Evaluation Metrics

The experimental workflow evaluates four distinct model architectures:
- Multiple Linear Regression (MLR): Serves as the benchmark baseline.
- Random Forest Regression (RF): Evaluates non-linear bagging ensemble capabilities.
- CatBoost Regression: Evaluates gradient boosting performance on structured tabular data.
- Multilayer Perceptron (MLP): Evaluates feedforward artificial neural network representations.

Evaluation criteria:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R-squared)

## 8. Formation Pressure Reference Data and Split Audit

Official formation pressure measurements were extracted and reconciled across three primary sources for well 15/9-F-12: the Schlumberger MWD Stethoscope pretest report (FM_PRESS_COMPUTED_MWD_1.ASC), the high-precision Quartz Gauge (AQAP) certification report (Test point table.xls), and the official well completion records.

Key quality and dataset audit resolutions:
- Test 14 TVD corrected: Fixed erroneous TVD entry (2,970.40 m TVD RKB / 2,915.50 m TVD MSL).
- Test 19 recovered: Recovered high-precision memory quartz gauge record (357.714 bar at 3,403.53 m MD / 3,036.23 m TVD RKB).
- Pressure test lithology: 22 of 23 test stations reside in reservoir sandstone (SAND_FLAG = 1.0). Only station 17 is in tight shale/siltstone (SAND_FLAG = 0.0).
- Contiguous partition alignment: 15 stations reside in Training (MD <= 3,320.80 m), 2 in Validation (3,320.95 to 3,362.25 m), and 4 in Testing (3,362.40 to 3,403.57 m).

| Test No | Run | Formation | Depth MD (m) | TVD RKB (m) | TVD MSL (m) | Pressure (bar) | Split Partition | Quality Interpretation | Recommended Role |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1  | 10 | Hugin     | 3127.71 | 2871.76 | 2816.86 | 320.020 | Training   | Very good permeability (AQAP) | Primary calibration reference |
| 2  | 10 | Hugin     | 3127.99 | 2871.92 | 2817.02 | 320.042 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 3  | 10 | Hugin     | 3136.74 | 2877.09 | 2822.19 | 320.385 | Training   | Very good permeability (AQAP) | Primary calibration reference |
| 4  | 10 | Hugin     | 3144.07 | 2881.43 | 2826.53 | 320.713 | Training   | Very good permeability (AQAP) | Primary calibration reference |
| 5  | 10 | Hugin     | 3152.61 | 2886.49 | 2831.59 | 321.050 | Training   | Very good permeability (AQAP) | Primary calibration reference |
| 6  | 10 | Hugin     | 3170.73 | 2897.20 | 2842.30 | 321.827 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 7  | 10 | Hugin     | 3191.85 | 2909.70 | 2854.80 | 322.677 | Training   | Very good permeability (AQAP) | Primary calibration reference |
| 8  | 10 | Hugin     | 3221.68 | 2927.35 | 2872.45 | 323.892 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 9  | 10 | Hugin     | 3240.73 | 2938.67 | 2883.77 | 324.698 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 10 | 10 | Hugin     | 3254.85 | 2947.11 | 2892.21 | 325.298 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 12 | 10 | Hugin     | 3262.78 | 2951.85 | 2896.95 | 325.686 | Training   | Poor permeability (AQAP)      | Secondary reference candidate |
| 11 | 10 | Hugin     | 3263.51 | 2952.29 | 2897.39 | 325.702 | Training   | Good permeability (AQAP)      | Primary calibration reference |
| 13 | 10 | Sleipner  | 3293.55 | 2970.18 | 2915.28 | 328.056 | Training   | Unstabilized drawdown         | Exclude from calibration |
| 14 | 10 | Sleipner  | 3294.00 | 2970.40 | 2915.50 | 334.101 | Training   | Poor permeability (AQAP)      | Secondary reference candidate |
| 15 | 10 | Sleipner  | 3307.73 | 2978.60 | 2923.70 | 335.610 | Training   | Poor permeability (AQAP)      | Secondary reference candidate |
| 16 | 10 | Sleipner  | 3324.65 | 2988.70 | 2933.80 | 341.713 | Validation | Good permeability (AQAP)      | Validation reference candidate |
| 17 | 10 | Sleipner  | 3333.57 | 2994.05 | 2939.15 | 69.279  | Validation | Tight formation / dry test    | Exclude from quantitative PP |
| 18 | 10 | Skagerrak | 3386.51 | 3025.98 | 2971.08 | 354.001 | Testing    | Fair permeability (AQAP)      | Secondary reference candidate |
| 20 | 10 | Skagerrak | 3386.99 | 3026.27 | 2971.37 | 368.706 | Testing    | Good permeability (AQAP)      | Final test reference candidate |
| 19 | 10 | Skagerrak | 3403.53 | 3036.23 | 2981.33 | 357.714 | Testing    | Good permeability (Memory)    | Final test reference candidate |
| 21 | 10 | Skagerrak | 3403.57 | 3036.26 | 2981.36 | 368.818 | Testing    | Good permeability (AQAP)      | External boundary reference |

Detailed audit findings:
- Validation partition contains exactly 1 good test (Test 16: 341.713 bar) and 1 dry test (Test 17: 69.279 bar).
- Testing partition contains 3 valid tests (Tests 18, 19, 20) and 1 external boundary station (Test 21 at 3,403.57 m MD, 0.02 m below predictor cutoff).
- An 11.10 bar discrepancy exists between Test 19 (357.714 bar at 3,403.53 m) and Test 21 (368.818 bar at 3,403.57 m), reflecting differing drawdown regimes or isolated sand compartments that must not be averaged.

## 9. Stage 2: Overburden Stress Profile (Sv) Reconstruction

Total vertical lithostatic stress Sv(z) represents the cumulative weight of overlying pore fluids and rock matrix per unit area from surface down to depth z:

Sv(z) = integral from 0 to z of rho(z') * g * dz'

### Reference Datums and Geometry
- Kelly Bushing Elevation (RKB): 54.90 m above Mean Sea Level (MSL).
- Sea Level (MSL): 0.0 m TVD MSL (54.90 m TVD RKB).
- Seabed / Mudline: 91.00 m TVD MSL (145.90 m TVD RKB).
- Air Gap (0.0 to 54.90 m RKB): Zero lithostatic contribution.
- Water Column (0.0 to 91.00 m TVD MSL): Central North Sea seawater density rho_w = 1.03 g/cm3, producing 9.1918 bar (133.32 psi) at seabed (gradient = 0.1010 bar/m).
- Shallow Sediment Interval (91.00 to 262.43 m TVD MSL): Thickness = 171.43 m.
- Continuous Measured Density Log: Decoded from WLC_COMPOSITE_1.LIS (262.43 m to 3,507.79 m MD), verified against WLC_PETRO_COMPUTED_INPUT_1.LAS (1,349.96 m to 3,507.49 m MD).

### Shallow Sediment Sensitivity Cases
Because continuous wireline logging commences at 262.43 m MD (Run 3 pilot hole ARC-ADN), the unlogged shallow sediment column between seabed (91.0 m TVD MSL) and 262.43 m TVD MSL (171.43 m thickness) is evaluated under three standard North Sea compaction cases:
- Low case: Mean density = 1.80 g/cm3 (stress contribution = 30.26 bar).
- Mid case (Nominal): Mean density = 1.88 g/cm3 (stress contribution = 31.61 bar).
- High case: Mean density = 1.95 g/cm3 (stress contribution = 32.78 bar).
Total shallow uncertainty is +/- 1.26 bar (+/- 0.21% of total overburden stress at reservoir depth).

### Overburden Interval Breakdown

| Interval Layer | Top MD (m) | Base MD (m) | Top TVD MSL (m) | Base TVD MSL (m) | Thickness (m) | Mean Density (g/cm3) | Layer Stress (bar) | Cumulative Sv (bar) | OBG (bar/m) | Equivalent Density (s.g.) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Air Gap | 0.00 | 54.90 | -54.90 | 0.00 | 54.90 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| Water Column (Seawater) | 54.90 | 145.90 | 0.00 | 91.00 | 91.00 | 1.030 | 9.1918 | 9.1918 | 0.1010 | 1.030 |
| Shallow Unlogged Sediments | 145.90 | 262.43 | 91.00 | 262.43 | 171.43 | 1.880 | 31.6061 | 40.7979 | 0.1555 | 1.585 |
| Upper Tertiary Claystones | 262.43 | 1350.00 | 262.43 | 1323.84 | 1061.41 | 2.011 | 204.3997 | 253.9302 | 0.1891 | 1.928 |
| Mid-Lower Tertiary Section | 1350.00 | 2500.00 | 1323.84 | 2412.27 | 1088.42 | 2.104 | 229.4927 | 474.6902 | 0.1968 | 2.007 |
| Upper Cretaceous Chalk Group | 2500.00 | 3127.71 | 2412.27 | 2853.25 | 440.99 | 2.470 | 105.3638 | 580.0540 | 0.2033 | 2.073 |
| Study / Predictor Interval | 3127.71 | 3403.55 | 2853.25 | 3017.83 | 164.58 | 2.355 | 38.0167 | 618.0707 | 0.2048 | 2.088 |

### Overburden Stress Summary across Study Interval
- Top of Predictor Interval (3,127.71 m MD / 2,853.25 m TVD MSL):
  - Measured log contribution: 539.26 bar
  - Shallow sediment contribution (Mid): 31.61 bar (Low: 30.26 bar, High: 32.78 bar)
  - Seawater column contribution: 9.19 bar
  - Total vertical stress Sv (Mid): 580.05 bar (8,413 psi / 58.01 MPa)
  - Overburden gradient (OBG): 0.2033 bar/m (2.073 s.g. / 0.899 psi/ft / 17.30 ppg)
- Base of Predictor Interval (3,403.55 m MD / 3,017.83 m TVD MSL):
  - Measured log contribution: 577.27 bar
  - Total vertical stress Sv (Mid): 618.07 bar (8,964 psi / 61.81 MPa)
  - Overburden gradient (OBG): 0.2048 bar/m (2.088 s.g. / 0.905 psi/ft / 17.43 ppg)
- Effective vertical stress in reservoir sands (sigma_v = Sv - P_fm):
  - Hugin Formation (Tests 1 to 12): 260.03 to 272.14 bar
  - Sleipner Formation (Tests 14 to 16): 264.78 to 268.47 bar
  - Skagerrak Formation (Tests 18 to 21): 246.99 to 261.63 bar

The overburden reconstruction profile is stored in [overburden_stress_profile.csv](file:///Users/user/final-year-project/tables/overburden_stress_profile.csv) and [overburden_stress_profile.json](file:///Users/user/final-year-project/tables/overburden_stress_profile.json).
The interval layer breakdown is stored in [overburden_interval_breakdown.csv](file:///Users/user/final-year-project/tables/overburden_interval_breakdown.csv) and [overburden_interval_breakdown.json](file:///Users/user/final-year-project/tables/overburden_interval_breakdown.json).
The stress state at all 21 pressure test stations is stored in [overburden_at_pressure_stations.csv](file:///Users/user/final-year-project/tables/overburden_at_pressure_stations.csv) and [overburden_at_pressure_stations.json](file:///Users/user/final-year-project/tables/overburden_at_pressure_stations.json).

## 10. Hydrostatic Reference Profiles and Reservoir Fluid Regimes

A rigorous hydrostatic reference requires separating the open seawater column above the seabed from the formation fluid column below the mudline, and accounting for reservoir fluid density regimes across the three target formations:

### Fluid Density and Pressure Regimes
1. Water Column (0 to 91.0 m TVD MSL):
   - Seawater density rho_w = 1.025 to 1.030 g/cm3.
   - Hydrostatic pressure at seabed = 9.147 to 9.192 bar (132.67 to 133.32 psi).
2. Regional Formation Brine Hydrostatic (Seabed to Reservoir Depth):
   - Normal Central North Sea formation water density: 1.025 to 1.050 g/cm3 (gradients: 0.1005 to 0.1030 bar/m).
   - Regional brine hydrostatic pressure at 2,872 m TVD MSL is approximately 307.0 to 308.4 bar.
   - Measured Hugin sandstone formation pressures (320.02 to 325.70 bar) indicate an overpressure of 11.6 to 13.1 bar above regional brine hydrostatic.
3. Hugin Formation Hydrocarbon Regime (Tests 1 to 12):
   - Linear fit: Pressure = 118.20 + 0.07028 * TVD (R2 = 0.999915, MAE = 0.017 bar).
   - Fluid gradient = 0.07028 bar/m, corresponding to an in situ oil density of 0.7167 g/cm3 (42.6 deg API).
   - Confirms a single, continuously communicating hydrocarbon column across the entire Hugin Sandstone interval.
4. Sleipner Transitional Regime (Tests 13 to 16):
   - Pressures increase step-wise from 328.06 bar (Test 13) to 341.71 bar (Test 16).
   - Apparent gradient = 0.5786 bar/m (R2 = 0.8139), reflecting structural compartmentalization and low-permeability mudstone baffles between individual sand layers.
5. Skagerrak Compartmentalized Regime (Tests 18 to 21):
   - Extreme near-depth pressure discrepancies confirm distinct compartment boundaries or localized pretest drawdowns:
     - Depth pair 1 (3,026 m TVD): Test 18 (fair test, 354.001 bar) versus Test 20 (good test, 368.706 bar) exhibits a 14.705 bar discrepancy across only 0.29 m TVD.
     - Depth pair 2 (3,036 m TVD): Test 19 (memory quartz gauge, 357.714 bar) versus Test 21 (realtime AQAP gauge, 368.818 bar) exhibits an 11.104 bar discrepancy across only 0.03 m TVD.
   - These measurements cannot be treated as a continuous uniform hydrostatic gradient and must not be averaged.

### Reconciliation of Overburden Sensitivity
In ChatGPT's provisional Stage 2 evaluation, relying exclusively on WLC_PETRO_COMPUTED_INPUT_1.LAS (density start at 1,349.96 m MD) suggested 1,362 m of missing shallow sediment and generated a 53.44 bar sensitivity spread (Sv = 548.00 to 601.44 bar at top; 585.88 to 639.32 bar at base).

Decoding the complete composite logging archive (WLC_COMPOSITE_1.LIS and WL_RAW_CAL-DEN-GR-NEU-REMP_MWD_1.DLIS from Run 3 pilot hole ARC-ADN) resolves this limitation:
- Continuous measured LWD bulk density is available starting at 262.43 m MD (262.43 m TVD MSL).
- The empirical mean bulk density between 262.43 m and 1,350.0 m MD is 2.011 g/cm3.
- The actual unlogged interval is restricted to only 171.43 m (seabed 91.0 m to 262.43 m TVD MSL).
- This establishes that the true overburden profile lies between ChatGPT's 1.90 g/cm3 and 2.00 g/cm3 sensitivity scenarios:
  - Predictor top (3,127.71 m MD): Sv = 580.05 bar (OBG = 0.2033 bar/m / 2.073 s.g.).
  - Predictor base (3,403.55 m MD): Sv = 618.07 bar (OBG = 0.2048 bar/m / 2.088 s.g.).
  - The uncertainty spread is collapsed from 53.44 bar down to +/- 1.26 bar (+/- 0.21%).

The hydrostatic and fluid regime tables are stored in:
- [hydrostatic_sensitivity_1811.csv](file:///Users/user/final-year-project/tables/hydrostatic_sensitivity_1811.csv) and [hydrostatic_sensitivity_1811.json](file:///Users/user/final-year-project/tables/hydrostatic_sensitivity_1811.json).
- [hydrostatic_pressure_test_comparison.csv](file:///Users/user/final-year-project/tables/hydrostatic_pressure_test_comparison.csv) and [hydrostatic_pressure_test_comparison.json](file:///Users/user/final-year-project/tables/hydrostatic_pressure_test_comparison.json).
- [pressure_fluid_gradient_reconciliation.csv](file:///Users/user/final-year-project/tables/pressure_fluid_gradient_reconciliation.csv) and [pressure_fluid_gradient_reconciliation.json](file:///Users/user/final-year-project/tables/pressure_fluid_gradient_reconciliation.json).
- [pressure_regime_summary.csv](file:///Users/user/final-year-project/tables/pressure_regime_summary.csv) and [pressure_regime_summary.json](file:///Users/user/final-year-project/tables/pressure_regime_summary.json).
- [near_depth_pressure_pairs.csv](file:///Users/user/final-year-project/tables/near_depth_pressure_pairs.csv) and [near_depth_pressure_pairs.json](file:///Users/user/final-year-project/tables/near_depth_pressure_pairs.json).
