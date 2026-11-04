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

## 8. Target Variable Status

The target variable is Pore Pressure (PP). Training and final validation are pending final target integration:
- Primary pathway: direct confirmation from author correspondence.
- Secondary pathway: derivation from official Equinor formation pressure data.
