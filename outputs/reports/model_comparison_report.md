# Machine Learning Model Training, Evaluation, and Comparison Report
## Formation Pore-Pressure Prediction in Well 15/9-F-12 (Volve Field)

**Document Reference:** `outputs/reports/model_comparison_report.md`  
**Execution Baseline Commit:** `d0ea70d`  
**Evaluation Date:** 2026-11-11  
**Target Variable:** Formation Pore Pressure ($P_p$, bar)  
**Total Observations:** 1,811 (1,414 valid targets; 397 excluded transition-zone observations)  

---

## Executive Summary

This report delivers a rigorous comparative evaluation of five candidate machine learning regression models for pore-pressure prediction in Well 15/9-F-12:
1. **Multiple Linear Regression (MLR)** (baseline parametric model)
2. **Random Forest Regressor (RF)** (ensemble bagging)
3. **CatBoost Regressor** (symmetric gradient boosting)
4. **Gradient Boosting Regressor (GBR)** (greedy stagewise gradient boosting)
5. **Multi-Layer Perceptron (MLP)** (feedforward artificial neural network)

### Key Conclusions
- **Training Set Fidelity:** Tree ensembles achieved outstanding fit on the 1,268 training rows (CatBoost RMSE: 0.202 bar, $R^2 = 0.999$; Gradient Boosting RMSE: 0.217 bar, $R^2 = 0.999$; Random Forest RMSE: 0.332 bar, $R^2 = 0.997$).
- **Intermediate Validation Generalization (Test 16 Checkpoint):** The Multi-Layer Perceptron (MLP) demonstrated superior out-of-interval generalization in the lower Sleipner Formation, predicting Test 16 at **341.44 bar** against the physical measured value of **341.713 bar** (an absolute error of only **0.270 bar**). By contrast, tree models (Random Forest, CatBoost, Gradient Boosting) and MLR were constrained by training label limits and predicted 334.80 to 335.82 bar (residuals of +5.89 to +6.91 bar).
- **Physical Extrapolation to the Overpressured Skagerrak Regime (Tests 20 & 21):** All five models exhibited an underprediction bias of 28.7 to 38.9 bar in the deep Skagerrak Formation. Because decision trees mathematically cannot predict values outside the bounding box of their training labels ($[320.02, 336.43]$ bar), tree models saturated near 336.5 to 340.1 bar when encountering the 368.71 to 368.82 bar reservoir compartment.
- **Ensemble Justification:** A stacking ensemble is explicitly **not recommended** because all models share the same fundamental physical constraint: logging tools alone in this vertical sequence do not encode the structural pressure compartment offset across the unmeasured transition seal without incorporating depth memorization or physics-based regional compartmentalization priors.
- **Recommended Primary Model:** **CatBoost Regressor** is recommended for reservoir intervals within calibrated pressure regimes due to its superior numerical stability, lowest training RMSE (0.202 bar), and balanced feature interpretation. For uncalibrated transition extrapolation where bounded linear/continuous response is required, **MLP Regressor** serves as an indispensable physics-complementary architecture.

---

## 1. Data Integrity and Anti-Leakage Protocol

The modeling pipeline strictly adheres to five core data integrity constraints:

1. **Contiguous Depth Partitioning (70:15:15):**
   - **Training Set:** MD 3,127.7052 m to 3,320.7960 m (1,268 rows, 100% valid targets). Covers the Hugin Sandstone and upper-to-middle Sleipner Formation ($P_p \in [320.02, 336.43]$ bar).
   - **Validation Set:** MD 3,320.9484 m to 3,362.2488 m (272 rows: 34 valid lower Sleipner targets, 238 invalid transition targets). Contains unseen validation checkpoint **Test 16** (341.713 bar).
   - **Testing Set:** MD 3,362.4012 m to 3,403.5492 m (271 rows: 159 invalid transition targets, 112 valid Skagerrak targets). Contains unseen physical test stations **Test 20** (368.706 bar) and **Test 21** (368.818 bar).

2. **Strict Predictor Feature Set:**
   Only the 13 verified petrophysical and drilling logs were provided to the models:
   `["DT", "GR", "NPHI", "RD", "RHOB", "RS", "RT", "ROP", "BVW", "K", "PHIF", "SW", "VCL"]`.
   `DEPTH`, `TVD_MSL`, `TVD_RKB`, `TARGET_PROVENANCE`, `MEASURED_TEST_NO`, `MEASURED_PRESSURE_BAR`, and target-derived variables were strictly excluded from the predictor matrix.

3. **Strict Preprocessing Isolation:**
   - Log10 transformations on skewed logs (`K, RS, RD, RT`) and `StandardScaler` transformations were fitted **exclusively on the 1,268 Training observations**.
   - No validation or testing observations were used to fit preprocessing scalers or model parameters.
   - For CatBoost and Gradient Boosting, early stopping was disabled or restricted strictly to internal training splits to guarantee zero validation leakage.

4. **Exclusion of Unsupported Transition Intervals:**
   The 397 rows across MD 3,326.0 m to 3,386.5 m lack physical pressure calibration and were flagged as `IS_TARGET_VALID == False`. They were strictly excluded from training loss calculations and primary performance metrics.

---

## 2. Global and Partition Performance Evaluation

Performance metrics are reported separately across partitions in Table 1.

### Table 1: Model Performance Breakdown by Partition (Valid Targets Only)

| Model | Partition Subset | N Valid | MAE (bar) | RMSE (bar) | $R^2$ | Mean Residual (bar) | MaxAE (bar) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLR** | Training (Valid) | 1,268 | 1.2194 | 1.9334 | 0.8888 | 0.0000 | 10.6501 |
| **MLR** | Validation (Valid) | 34 | 5.9796 | 5.9925 | -4,418.07 | +5.9796 | 6.9186 |
| **MLR** | Testing (Tests 20 & 21 Anchors) | 2 | 36.8713 | 36.9302 | -434,897.92 | +36.8713 | 38.9578 |
| **MLR** | Testing (Provisional Assumption) | 110 | 36.1347 | 36.1885 | N/A | +36.1347 | 40.2719 |
| **MLR** | Full Dataset (All Valid) | 1,414 | 4.1005 | 10.3934 | 0.3417 | +3.0070 | 40.2719 |
| **Random Forest** | Training (Valid) | 1,268 | 0.1481 | 0.3319 | 0.9967 | -0.0026 | 3.4656 |
| **Random Forest** | Validation (Valid) | 34 | 6.5875 | 6.6139 | -5,382.03 | +6.5875 | 7.5489 |
| **Random Forest** | Testing (Tests 20 & 21 Anchors) | 2 | 32.0953 | 32.1223 | -329,029.45 | +32.0953 | 33.4108 |
| **Random Forest** | Testing (Provisional Assumption) | 110 | 33.1743 | 33.1948 | N/A | +33.1743 | 34.4112 |
| **Random Forest** | Full Dataset (All Valid) | 1,414 | 2.9174 | 9.3984 | 0.4617 | +2.7822 | 34.4112 |
| **CatBoost** | Training (Valid) | 1,268 | **0.1395** | **0.2023** | **0.9988** | +0.0004 | **1.3851** |
| **CatBoost** | Validation (Valid) | 34 | 6.3066 | 6.3345 | -4,936.88 | +6.3066 | 7.6325 |
| **CatBoost** | Testing (Tests 20 & 21 Anchors) | 2 | 31.1577 | 31.1652 | -309,714.62 | +31.1577 | 31.8388 |
| **CatBoost** | Testing (Provisional Assumption) | 110 | 33.5800 | 33.5942 | N/A | +33.5800 | 35.4199 |
| **CatBoost** | Full Dataset (All Valid) | 1,414 | 2.9332 | 9.4958 | 0.4505 | +2.8083 | 35.4199 |
| **Gradient Boosting**| Training (Valid) | 1,268 | 0.1457 | 0.2168 | 0.9986 | -0.0002 | 1.7217 |
| **Gradient Boosting**| Validation (Valid) | 34 | 6.3357 | 6.3508 | -4,962.32 | +6.3357 | 7.5999 |
| **Gradient Boosting**| Testing (Tests 20 & 21 Anchors) | 2 | **30.4076** | **30.4558** | -295,776.35 | +30.4076 | 32.1205 |
| **Gradient Boosting**| Testing (Provisional Assumption) | 110 | 33.0321 | 33.0979 | N/A | +33.0321 | 35.0299 |
| **Gradient Boosting**| Full Dataset (All Valid) | 1,414 | **2.8957** | 9.3565 | 0.4665 | +2.7649 | 35.0299 |
| **MLP** | Training (Valid) | 1,268 | 0.9936 | 1.4951 | 0.9335 | -0.0348 | 11.1011 |
| **MLP** | Validation (Valid) | 34 | **2.7071** | **3.5882** | **-1,583.43** | **+2.2677** | 8.2514 |
| **MLP** | Testing (Tests 20 & 21 Anchors) | 2 | 31.1097 | 31.1926 | -310,260.15 | +31.1097 | 33.3831 |
| **MLP** | Testing (Provisional Assumption) | 110 | **31.2473** | **31.4296** | N/A | **+31.2473** | 36.8874 |
| **MLP** | Full Dataset (All Valid) | 1,414 | 3.4309 | **8.9742** | **0.5092** | +2.4981 | 36.8874 |

### Mathematical Note on Negative $R^2$ on Narrow Depth Intervals
In the Validation partition ($n=34$), the target pressure varies over a very narrow range (341.135 to 341.796 bar), yielding a near-zero Total Sum of Squares:
$$SS_{\text{tot}} = \sum_{i=1}^{34} (y_i - \bar{y})^2 \approx 0.458\text{ bar}^2$$
A residual error of approximately 6 bar produces a Residual Sum of Squares ($SS_{\text{res}}$) of over $1,220\text{ bar}^2$. Consequently:
$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{1,220}{0.458} \approx -2,660$$
In the Test partition, target variance across the 110 provisional rows is identically zero ($SS_{\text{tot}} = 0$). Hence, $R^2$ is undefined (NaN) or massively negative. Therefore, $R^2$ is mathematically degenerate over narrow constant-pressure compartments; MAE, RMSE, and physical station residuals provide the only scientifically valid metrics.

---

## 3. Evaluation by Target Provenance

Table 2 breaks down performance across target provenance categories:
1. `measured_anchor`: Discrete Schlumberger wireline FPWD pretests ($n=17$ total: 14 in Train, 1 in Val, 2 in Test).
2. `physics_derived`: Continuous hydrodynamic gradients ($n=1,287$ total: 1,254 in Train, 33 in Val).
3. `provisional_assumption`: Constant-compartment Skagerrak assumption ($n=110$ in Test).

### Table 2: Model Performance Breakdown by Target Provenance

| Model | Target Provenance | Partition | N Obs | MAE (bar) | RMSE (bar) | $R^2$ | Mean Residual (bar) | MaxAE (bar) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLR** | `measured_anchor` | Training | 14 | 0.8121 | 0.9829 | 0.9568 | -0.3144 | 1.8264 |
| **MLR** | `measured_anchor` | Validation (Test 16) | 1 | 5.8915 | 5.8915 | N/A | +5.8915 | 5.8915 |
| **MLR** | `measured_anchor` | Testing (Tests 20 & 21)| 2 | 36.8713 | 36.9302 | -434,897.92 | +36.8713 | 38.9578 |
| **MLR** | `physics_derived` | Training | 1,254 | 1.2239 | 1.9414 | 0.8881 | +0.0035 | 10.6501 |
| **MLR** | `physics_derived` | Validation | 33 | 5.9822 | 5.9955 | -4,369.82 | +5.9822 | 6.9186 |
| **MLR** | `provisional_assumption` | Testing | 110 | 36.1347 | 36.1885 | N/A | +36.1347 | 40.2719 |
| **Random Forest** | `measured_anchor` | Training | 14 | 0.1927 | 0.4980 | 0.9889 | -0.1878 | 1.8218 |
| **Random Forest** | `measured_anchor` | Validation (Test 16) | 1 | 6.9122 | 6.9122 | N/A | +6.9122 | 6.9122 |
| **Random Forest** | `measured_anchor` | Testing (Tests 20 & 21)| 2 | 32.0953 | 32.1223 | -329,029.45 | +32.0953 | 33.4108 |
| **Random Forest** | `physics_derived` | Training | 1,254 | 0.1476 | 0.3296 | 0.9968 | -0.0005 | 3.4656 |
| **Random Forest** | `physics_derived` | Validation | 33 | 6.5777 | 6.6046 | -5,303.01 | +6.5777 | 7.5489 |
| **Random Forest** | `provisional_assumption` | Testing | 110 | 33.1743 | 33.1948 | N/A | +33.1743 | 34.4112 |
| **CatBoost** | `measured_anchor` | Training | 14 | **0.1452** | **0.2683** | **0.9968** | -0.0996 | **0.9418** |
| **CatBoost** | `measured_anchor` | Validation (Test 16) | 1 | 6.4804 | 6.4804 | N/A | +6.4804 | 6.4804 |
| **CatBoost** | `measured_anchor` | Testing (Tests 20 & 21)| 2 | 31.1577 | 31.1652 | -309,714.62 | +31.1577 | 31.8388 |
| **CatBoost** | `physics_derived` | Training | 1,254 | **0.1395** | **0.2014** | **0.9988** | +0.0015 | **1.3851** |
| **CatBoost** | `physics_derived` | Validation | 33 | 6.3013 | 6.3300 | -4,871.13 | +6.3013 | 7.6325 |
| **CatBoost** | `provisional_assumption` | Testing | 110 | 33.5800 | 33.5942 | N/A | +33.5800 | 35.4199 |
| **Gradient Boosting**| `measured_anchor`| Training | 14 | 0.1884 | 0.3167 | 0.9955 | -0.1650 | 0.9742 |
| **Gradient Boosting**| `measured_anchor`| Validation (Test 16) | 1 | 6.2552 | 6.2552 | N/A | +6.2552 | 6.2552 |
| **Gradient Boosting**| `measured_anchor`| Testing (Tests 20 & 21)| 2 | **30.4076** | **30.4558** | -295,776.34 | +30.4076 | 32.1205 |
| **Gradient Boosting**| `physics_derived`| Training | 1,254 | 0.1452 | 0.2154 | 0.9986 | +0.0017 | 1.7217 |
| **Gradient Boosting**| `physics_derived`| Validation | 33 | 6.3381 | 6.3537 | -4,907.61 | +6.3381 | 7.5999 |
| **Gradient Boosting**| `provisional_assumption`| Testing | 110 | 33.0321 | 33.0979 | N/A | +33.0321 | 35.0299 |
| **MLP** | `measured_anchor` | Training | 14 | 0.8081 | 1.0444 | 0.9512 | -0.6062 | 2.2567 |
| **MLP** | `measured_anchor` | Validation (Test 16) | 1 | **0.2700** | **0.2700** | N/A | **+0.2700** | **0.2700** |
| **MLP** | `measured_anchor` | Testing (Tests 20 & 21)| 2 | 31.1097 | 31.1926 | -310,260.15 | +31.1097 | 33.3831 |
| **MLP** | `physics_derived` | Training | 1,254 | 0.9956 | 1.4993 | 0.9333 | -0.0285 | 11.1011 |
| **MLP** | `physics_derived` | Validation | 33 | **2.7809** | **3.6419** | -1,611.72 | **+2.3282** | 8.2514 |
| **MLP** | `provisional_assumption` | Testing | 110 | **31.2473** | **31.4296** | N/A | **+31.2473** | 36.8874 |

---

## 4. Discrete Physical Station Benchmarks

Table 3 provides station-by-station evaluations focusing on discrete Schlumberger pretests.

### Table 3: Physical Station Benchmarks Across All Candidate Models

| Metric / Benchmark Station | Actual (bar) | MLR | Random Forest | CatBoost | Gradient Boosting | MLP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **15 Training Stations RMSE** | -- | 0.989 bar | 0.481 bar | **0.283 bar** | 0.307 bar | 1.016 bar |
| **Test 1 (3,127.71 m MD, Hugin)** | 320.020 | 321.846 (-1.826) | 320.185 (-0.165) | 320.171 (-0.151) | 320.236 (-0.216) | 320.144 (-0.124) |
| **Test 13 (Reconciled FPWD Plot)** | 334.161 | 335.236 (-1.075) | 334.095 (+0.066) | 333.725 (+0.436) | 334.052 (+0.109) | 334.635 (-0.474) |
| **Test 14 (3,294.00 m MD, Sleipner)** | 334.101 | 335.714 (-1.613) | 334.112 (-0.011) | 334.183 (-0.082) | 334.189 (-0.088) | 334.194 (-0.093) |
| **Test 15 (3,307.73 m MD, Sleipner)** | 335.610 | 335.609 (+0.001) | 335.587 (+0.023) | 335.727 (-0.117) | 335.630 (-0.020) | 337.505 (-1.895) |
| **Test 16 Checkpoint (3,324.65 m MD)**| **341.713** | 335.822 (+5.892) | 334.801 (+6.912) | 335.233 (+6.480) | 335.458 (+6.255) | **341.443 (+0.270)** |
| **Test 20 (3,386.99 m MD, Skagerrak)**| **368.706** | 333.921 (+34.785)| 335.295 (+33.411)| 336.867 (+31.839)| 336.586 (+32.121)| 335.323 (+33.383)|
| **Test 21 (3,403.57 m MD, Skagerrak)**| **368.818** | 329.860 (+38.958)| 338.038 (+30.780)| 338.341 (+30.477)| 340.123 (+28.695)| 339.982 (+28.836)|
| **Tests 20 & 21 Combined MAE** | -- | 36.871 bar | 32.095 bar | 31.158 bar | **30.408 bar** | 31.110 bar |
| **Tests 20 & 21 Combined RMSE** | -- | 36.930 bar | 32.122 bar | 31.165 bar | **30.456 bar** | 31.193 bar |
| **Excluded Test 19 (Memory Value)** | **357.714** | 329.860 (+27.854)| 338.038 (+19.676)| 338.341 (+19.373)| 340.123 (+17.591)| 339.982 (+17.732)|

*(Note: Parentheses indicate residual = Actual minus Predicted in bar).*

---

## 5. Model Diagnostics and Residual Analysis

### 5.1 Parity and Extrapolation Behavior
Inspection of `outputs/figures/actual_vs_predicted.png` demonstrates two distinct behaviors:
1. **In-Distribution Regimes (320 to 336 bar):** All models follow the 1:1 parity line accurately. CatBoost, Gradient Boosting, and Random Forest achieve near-zero dispersion.
2. **Out-of-Distribution Overpressure (341 to 369 bar):**
   - At the **341.71 bar** level (lower Sleipner), tree models flatten at their training ceiling (~335 bar). In contrast, the **MLP** accurately predicts 341.44 bar, tracking the step change.
   - At the **368.8 bar** level (Skagerrak), tree models saturate at 335.3 to 340.1 bar. The MLP also underpredicts at ~337.5 bar because petrophysical log responses in this zone do not provide sufficient unbounded signal to trigger a further 30 bar jump without explicit depth coordinates.

### 5.2 Continuous Borehole Pressure Profiles
In `outputs/figures/borehole_pressure_profiles.png`:
- The predicted pressure profiles remain between the normal hydrostatic gradient ($1.025\text{ g/cm}^3$, 284 to 305 bar) and the total lithostatic overburden stress $S_v$ (561 to 618 bar), satisfying fundamental physical bounds.
- The tree models exhibit high-frequency log-scale sensitivity within the reservoir sandstone but completely level off across the lower boundary.

### 5.3 Learning Curves and Convergence Diagnostics
In `outputs/figures/learning_curves.png`:
- CatBoost and Gradient Boosting achieve smooth, monotonic convergence on the training set without erratic oscillations.
- Validation loss on out-of-sample lower Sleipner stabilizes after ~100 iterations, confirming that overtraining on the training partition did not degrade extrapolation capability.
- MLP trained with the Adam optimizer converged within 836 epochs with early stopping.

---

## 6. Feature Importance and Petrophysical Interpretation

Feature importance was analyzed using both native split/gain metrics and model-independent permutation importance ($N=15$ repeats on the training partition). Detailed numerical values are preserved in `outputs/metrics/feature_importance.csv` and visualized in `outputs/figures/feature_importance_comparison.png`.

### Table 4: Feature Importance Ranking Summary

| Feature | Log Description | Tree Native Gain (CatBoost %) | Permutation Importance (% of RMSE Increase) | Petrophysical Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **ROP** | Rate of Penetration | **55.39%** | **40.16% (MLR), 38.77% (RF), 43.97% (CB), 61.37% (GB)** | Drilling penetration rate directly reflects the effective stress and chip-hold-down effect under varying bottom-hole overbalance. |
| **RD / RT** | Deep / True Resistivity | 10.12% / 2.74% | 16.23% (CB), 23.00% (RF) / 12.84% (MLP) | Resistivity logs capture fluid saturation, salinity, and compaction state, serving as classic overpressure indicators. |
| **SW / PHIF** | Water Sat / Porosity | 2.15% / 2.18% | 36.17% (MLP SW), 16.09% (MLP PHIF) | Key petrophysical variables driving the neural network's non-linear response across changing sand bodies. |
| **RS** | Shallow Resistivity | 6.62% | 7.48% (CB), 7.05% (RF) | Reflects mud-filtrate invasion dynamics and permeability contrast. |
| **VCL / GR** | Clay Volume / Gamma Ray | 5.06% / 2.77% | 4.26% (CB), 6.05% (MLP GR) | Discriminate reservoir sandstones from sealing shale/siltstone lithologies. |
| **RHOB** | Bulk Density | 1.94% | 3.14% (CB), 3.70% (MLP) | Direct measure of compaction and matrix density. |
| **DT** | Acoustic Slowness | 2.77% | 3.49% (CB), 1.69% (RF) | Sonic transit time responds to compaction, though partially eclipsed by ROP in tree splits. |

---

## 7. Model Selection, Recommendation, and Operational Limitations

### 7.1 Selection Matrix

| Evaluation Criterion | MLR | Random Forest | CatBoost | Gradient Boosting | MLP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Precision ($R^2$ / RMSE)** | Moderate (0.889 / 1.93 bar) | Very High (0.997 / 0.33 bar) | **Highest (0.999 / 0.20 bar)** | Very High (0.999 / 0.22 bar) | High (0.934 / 1.50 bar) |
| **Validation Generalization (Test 16)** | Poor (+5.89 bar) | Poor (+6.91 bar) | Poor (+6.48 bar) | Poor (+6.26 bar) | **Exceptional (+0.27 bar)** |
| **Test Anchors Absolute Error** | 36.87 bar | 32.10 bar | 31.16 bar | **30.41 bar** | 31.11 bar |
| **Sensitivity to Outliers / Noise** | High | Low | **Lowest** | Moderate | Moderate |
| **Petrophysical Interpretability** | High (Linear Coefs) | Moderate | **High (Gain + Permutation)**| Moderate | Low (Black Box) |

### 7.2 Why Stacking is Not Recommended
A meta-model stacking ensemble was considered and rejected. Because all five candidate models share the same predictor feature space (excluding depth), they all saturate between 335 and 340 bar in the deep Skagerrak. Stacking or linearly blending these models cannot extrapolate to 368.8 bar without introducing arbitrary, uncalibrated bias.

### 7.3 Primary Recommendation
1. **Primary Model for Calibrated Formations:** **CatBoost Regressor** is selected as the primary production model. It provides the highest accuracy across calibrated reservoir sands (RMSE = 0.202 bar, 15-station RMSE = 0.283 bar), robust tree-splitting stability, and consistent feature importance profiles.
2. **Secondary Complementary Model for In-Reservoir Step Transitions:** **MLP Regressor** is designated as the primary physics-complementary architecture. Its continuous non-linear mapping proved capable of predicting the lower Sleipner step change (+0.27 bar residual at Test 16).
3. **Operational Precondition for Deep Compartment Prediction:** For predicting completely unseen, structurally isolated overpressured regimes like the Skagerrak, machine learning models must be coupled with an explicit regional hydrodynamic compartment model or minimum-curvature geomechanical trend rather than relying solely on local well logs.

---

## 8. Artifact and Deliverable Manifest

All deliverables have been generated, dual-saved, and verified:

- **Serialized Model Artifacts:**
  - `outputs/models/mlr_model.joblib`
  - `outputs/models/random_forest_model.joblib`
  - `outputs/models/catboost_model.joblib`
  - `outputs/models/gradient_boosting_model.joblib`
  - `outputs/models/mlp_model.joblib`
  - `outputs/models/log_standard_scaler.joblib`
  - `outputs/models/training_manifest.json`
- **Dual-Saved Metrics Tables (CSV & JSON):**
  - `outputs/metrics/model_predictions_1811.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/partition_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/provenance_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/formation_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/station_evaluation_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/physical_station_benchmarks.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/feature_importance.csv` & `.json` (mirrored in `tables/`)
- **Diagnostic Figures:**
  - `outputs/figures/actual_vs_predicted.png`
  - `outputs/figures/residuals_vs_depth_and_dist.png`
  - `outputs/figures/borehole_pressure_profiles.png`
  - `outputs/figures/feature_importance_comparison.png`
  - `outputs/figures/learning_curves.png`
- **Automated Verification:**
  - `tests/test_model_training.py` (all 33 test cases passing with zero failures).
