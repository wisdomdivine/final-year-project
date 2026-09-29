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

| Dataset Partition | Percentage | Approximate Rows | Depth Interval (m) | Primary Purpose |
|:------------------|:-----------|:-----------------|:-------------------|:----------------|
| Training          | 70%        | 1,268            | 3127.71 to 3320.95 | Model parameter learning |
| Validation        | 15%        | 272              | 3321.10 to 3362.40 | Hyperparameter tuning and model selection |
| Testing           | 15%        | 271              | 3362.55 to 3403.55 | Final unbiased model evaluation |

## 4. Candidate Algorithms and Evaluation Metrics

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

## 5. Target Variable Status

The target variable is Pore Pressure (PP). Training and final validation are pending final target integration:
- Primary pathway: direct confirmation from author correspondence.
- Secondary pathway: derivation from official Equinor formation pressure data.
