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
- **Intermediate Validation Generalization (Test 16 Checkpoint):** The Multi-Layer Perceptron (MLP) demonstrated superior out-of-interval generalization in the lower Sleipner Formation, predicting Test 16 at **341.44 bar** against the physical measured value of **341.713 bar** (an absolute error of only **0.270 bar**). By contrast, tree models (Random Forest, CatBoost, Gradient Boosting) and MLR were constrained by training feature partitions and predicted 334.80 to 335.82 bar (residuals of +5.89 to +6.91 bar).
- **Physical Extrapolation and Mathematical Mechanics Across Architectures (Tests 20 & 21):** All five models exhibited an underprediction bias of 28.7 to 38.9 bar in the deep Skagerrak Formation. The underlying mathematical mechanisms differ fundamentally across model families:
  - *Random Forest (Averaging Ensemble):* Predicts via convex combinations of training labels across leaf nodes ($\hat{y} = \frac{1}{B}\sum_{b=1}^B \bar{y}_{\mathcal{L}_b}$). Its predictions are strictly bounded by the minimum and maximum of the training targets ($[320.020, 335.610]$ bar for measured anchors, or $[320.020, 336.428]$ bar for derived targets).
  - *Gradient Boosting and CatBoost (Additive Expansions):* Predict via staged sums of terminal leaves ($F_M(x) = F_0 + \sum_{m=1}^M \eta h_m(x)$). Because positive leaf outputs across trees accumulate additively, boosted trees can mathematically exceed the maximum training target (e.g., Gradient Boosting reached 340.123 bar on Test 21, and CatBoost reached 338.341 bar). However, outside the training feature support, decision rules plateau into constant step functions, preventing them from bridging the 30 bar tectonic compartment jump.
  - *Parametric (MLR) and Neural (MLP) Architectures:* Map continuous affine transformations ($w^T x + b$) and piecewise-linear ReLU paths, permitting unbounded linear extrapolation, but were limited here because logging tools alone in hard cemented sandstones do not reflect the isolated reservoir overpressure step without depth coordinates or compartment priors.
- **Model Selection Nuance:** No single architecture is universally superior. CatBoost achieves the lowest error and highest stability within calibrated in-distribution reservoir sandstones (14-station anchor RMSE = 0.268 bar; 15-station pretest RMSE = 0.283 bar). However, the Multi-Layer Perceptron (MLP) achieves the best validation generalization across uncalibrated stratigraphy (validation MAE = 2.707 bar; Test 16 error = +0.270 bar).
- **Ensemble Justification:** A stacking ensemble is explicitly **not recommended** because all models share the same fundamental physical constraint: logging tools alone in this vertical sequence do not encode the structural pressure compartment offset across the unmeasured transition seal without incorporating depth memorization or physics-based regional compartmentalization priors.


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

Performance must be evaluated with strict provenance separation:
1. **Measured Pretest Anchors (`measured_anchor`):** Discrete wireline pretest stations ($n=17$ total: 14 in Train, 1 in Val, 2 in Test). These constitute true ground truth.
2. **Physics-Derived Targets (`physics_derived`):** Continuous hydrodynamic pore-pressure gradients fitted from measured oil and brine gradients ($n=1,287$ total: 1,254 in Train, 33 in Val).
3. **Provisional Assumptions (`provisional_assumption`):** The 110 observations in the Skagerrak assigned a flat 368.762 bar value. **These must never be presented as physical ground truth or independent validation.** Model errors against these provisional values reflect deviation from a mathematical modeling assumption, not deviation from physical reality.


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

## 4. Discrete Physical Station Benchmarks and Station Reconciliation

### 4.1 Reconciliation of Training Stations: 14 Measured Anchors vs. 15 Pretest Stations
A critical audit distinction exists between the formal training anchor set and the full set of physical pretests located within the training interval:
1. **The 14 Primary Training Measured Anchors (`TARGET_PROVENANCE == 'measured_anchor'`):**
   - Exact station numbers: **Tests 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, and 15** ($n=14$).
   - These 14 stations stabilized with high physical quality during Schlumberger wireline pretests and directly anchored grid-node target values in the training set ($P_{\text{target}} = P_{\text{measured}}$).
   - RMSE across these 14 anchors: CatBoost = **0.268 bar**, Gradient Boosting = 0.317 bar, Random Forest = 0.498 bar, MLR = 0.983 bar, MLP = 1.044 bar.
2. **The 15 Total Pretest Stations in the Training Interval (Tests 1 through 15):**
   - Includes **Test 13** alongside the 14 anchors listed above ($n=15$).
   - **Test 13's Role:** Test 13 was an initial pretest aborted due to tool seal degradation. The preliminary Schlumberger summary table recorded 328.056 bar during an intermediate drawdown cycle. However, reconciliation of the original FPWD pressure-versus-time buildup plot (page 134 of the well report) established that the final formation buildup reached **334.161 bar** prior to tool retraction, closely matching companion repeat Test 14 (334.101 bar).
   - Because Test 13's grid target at row 1088 was generated from the middle-Sleipner hydrodynamic brine equation rather than pinned as a discrete measured anchor, its grid provenance is `physics_derived` (target = 334.074 bar, target residual = -0.087 bar).
   - When benchmarking models against physical pretest observations (using the reconciled 334.161 bar value for Test 13), the 15-station RMSE is: CatBoost = **0.283 bar**, Gradient Boosting = 0.307 bar, Random Forest = 0.481 bar, MLR = 0.989 bar, MLP = 1.016 bar.

### Table 3: Physical Station Benchmarks Across All Candidate Models

| Metric / Benchmark Station | Actual (bar) | MLR | Random Forest | CatBoost | Gradient Boosting | MLP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **14 Training Anchors RMSE (Tests 1-12, 14, 15)**| -- | 0.983 bar | 0.498 bar | **0.268 bar** | 0.317 bar | 1.044 bar |
| **15 Training Stations RMSE (Tests 1-15, incl. T13)**| -- | 0.989 bar | 0.481 bar | **0.283 bar** | 0.307 bar | 1.016 bar |
| **Test 1 (3,127.71 m MD, Hugin)** | 320.020 | 321.846 (-1.826) | 320.185 (-0.165) | 320.171 (-0.151) | 320.236 (-0.216) | 320.144 (-0.124) |
| **Test 2 (3,127.99 m MD, Hugin)** | 320.042 | 321.666 (-1.624) | 320.195 (-0.153) | 320.180 (-0.138) | 320.275 (-0.233) | 320.277 (-0.235) |
| **Test 3 (3,136.74 m MD, Hugin)** | 320.385 | 320.304 (+0.081) | 320.392 (-0.007) | 320.373 (+0.012) | 320.392 (-0.007) | 321.267 (-0.882) |
| **Test 4 (3,144.07 m MD, Hugin)** | 320.713 | 320.390 (+0.323) | 320.707 (+0.006) | 320.644 (+0.069) | 320.723 (-0.010) | 320.354 (+0.359) |
| **Test 5 (3,152.61 m MD, Hugin)** | 321.050 | 320.146 (+0.904) | 321.045 (+0.005) | 320.889 (+0.161) | 320.960 (+0.090) | 321.487 (-0.437) |
| **Test 6 (3,170.73 m MD, Hugin)** | 321.827 | 323.060 (-1.233) | 322.108 (-0.281) | 321.945 (-0.118) | 322.184 (-0.357) | 322.397 (-0.570) |
| **Test 7 (3,191.85 m MD, Hugin)** | 322.677 | 321.805 (+0.872) | 322.681 (-0.004) | 322.657 (+0.020) | 322.674 (+0.003) | 323.230 (-0.553) |
| **Test 8 (3,221.68 m MD, Hugin)** | 323.892 | 323.303 (+0.589) | 323.905 (-0.013) | 323.971 (-0.079) | 323.933 (-0.041) | 323.121 (+0.771) |
| **Test 9 (3,240.73 m MD, Hugin)** | 324.698 | 323.984 (+0.714) | 324.713 (-0.015) | 324.739 (-0.041) | 324.784 (-0.086) | 324.416 (+0.282) |
| **Test 10 (3,254.85 m MD, Hugin)** | 325.298 | 325.893 (-0.595) | 325.350 (-0.052) | 325.241 (+0.057) | 325.227 (+0.071) | 326.607 (-1.309) |
| **Test 11 (3,263.51 m MD, Hugin)** | 325.702 | 326.066 (-0.364) | 327.524 (-1.822) | 326.644 (-0.942) | 326.676 (-0.974) | 327.249 (-1.547) |
| **Test 12 (3,262.78 m MD, Hugin)** | 325.686 | 326.317 (-0.631) | 325.826 (-0.140) | 325.733 (-0.047) | 326.129 (-0.443) | 327.943 (-2.257) |
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

### 5.1 Parity and Mathematical Extrapolation Behavior
Inspection of `outputs/figures/actual_vs_predicted.png` demonstrates two distinct operational regimes:
1. **In-Distribution Reservoir Sequences (320 to 336 bar):**
   All models follow the 1:1 parity line accurately. CatBoost, Gradient Boosting, and Random Forest achieve near-zero dispersion on training anchors and physics-derived targets.
2. **Out-of-Distribution Geological Steps and Overpressure (341 to 369 bar):**
   - **Validation Step (Lower Sleipner, 341.71 bar):** The **MLP** accurately predicts 341.44 bar (residual +0.27 bar), demonstrating that its continuous representation can extrapolate non-linear shifts when driven by petrophysical log contrast. Conversely, tree models predict 334.80 to 335.82 bar (+5.89 to +6.91 bar residuals).
   - **Skagerrak Overpressure Regime (368.71 to 368.82 bar):**
     - *Random Forest:* Bounded by training target extremes ($[320.020, 336.428]$ bar) because its prediction is an unweighted average over terminal leaf samples ($\hat{y} = \frac{1}{B}\sum_{b=1}^B \bar{y}_{\mathcal{L}_b}$). It predicts 335.30 to 338.04 bar.
     - *CatBoost and Gradient Boosting:* Being additive expansions ($F_M(x) = F_0 + \sum_{m=1}^M \eta h_m(x)$), their predictions can accumulate positive leaf outputs and exceed the maximum training target. Empirically, Gradient Boosting outputs 340.123 bar on Test 21, and CatBoost outputs 338.341 bar. However, because tree splits are orthogonal step thresholds, once feature values exceed the training partition boundaries, predictions plateau into constant values rather than tracking physical gradients.
     - *MLP and MLR:* Exhibit continuous affine response curves, but because petrophysical logging tools in tight sandstone do not register the 30 bar hydraulic jump across the structural seal, they output ~330 to 340 bar without depth or compartment coordinates.


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

### 7.1 Selection Matrix and Architecture Trade-Offs

| Evaluation Criterion | MLR | Random Forest | CatBoost | Gradient Boosting | MLP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Fit ($R^2$ / RMSE)** | Moderate (0.889 / 1.93 bar) | Very High (0.997 / 0.33 bar) | **Highest (0.999 / 0.20 bar)** | Very High (0.999 / 0.22 bar) | High (0.934 / 1.50 bar) |
| **14-Anchor Calibration RMSE** | 0.983 bar | 0.498 bar | **0.268 bar** | 0.317 bar | 1.044 bar |
| **Validation Generalization (Test 16 Checkpoint)**| Poor (+5.89 bar) | Poor (+6.91 bar) | Poor (+6.48 bar) | Poor (+6.26 bar) | **Exceptional (+0.27 bar)** |
| **Validation Subset MAE (34 Rows)** | 5.980 bar | 6.588 bar | 6.307 bar | 6.336 bar | **2.707 bar** |
| **Test Anchors Error (Tests 20 & 21 MAE)** | 36.87 bar | 32.10 bar | 31.16 bar | **30.41 bar** | 31.11 bar |
| **Sensitivity to Outliers / Noise** | High | Low | **Lowest** | Moderate | Moderate |
| **Petrophysical Interpretability** | High (Linear Coefs) | Moderate | **High (Gain + Permutation)**| Moderate | Low (Black Box) |

### 7.2 Balanced Model Architecture Conclusion
No single model family emerges as universally superior across all evaluation regimes:
1. **In-Distribution Precision (CatBoost Advantage):**
   Within the calibrated reservoir sandstone sequences (Hugin and upper-to-middle Sleipner), **CatBoost** achieves the highest precision (training RMSE = 0.202 bar; 14-anchor calibration RMSE = 0.268 bar; 15-station pretest RMSE = 0.283 bar). Its symmetric trees offer numerical robustness and superior feature importance stability.
2. **Out-of-Interval Generalization (MLP Advantage):**
   Across the geological step transition into the lower Sleipner formation, the **Multi-Layer Perceptron (MLP)** exhibits superior out-of-interval generalization, predicting Test 16 with an error of only **+0.270 bar** and reducing validation MAE to **2.707 bar** (compared to 6.31 to 6.59 bar for tree models).
3. **Common Extrapolation Limitation:**
   Neither architecture alone resolves the deep Skagerrak compartment offset when trained solely on logs. Decision trees plateau into step functions, and neural networks underpredict by ~31 bar because petrophysical response curves alone do not encode tectonic overpressure across regional unconformities.

### 7.3 Why Stacking is Not Recommended
A meta-model stacking ensemble was considered and rejected. Because all five candidate models share the same predictor feature space (excluding depth), they all saturate between 335 and 340 bar in the deep Skagerrak. Stacking or linearly blending these models cannot extrapolate to 368.8 bar without introducing arbitrary, uncalibrated bias.


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
  - `tests/test_model_training.py` (all 33 baseline test cases passing with zero failures).

---

## 9. Controlled Experimental Comparisons

To systematically investigate whether non-petrophysical features can bridge the observed domain shifts without compromising geological integrity, four controlled experimental comparisons were conducted alongside the baseline:
1. **Experiment A (Depth-Aware):** Adds True Vertical Depth Subsea (`TVD_MSL`, m MSL) from Trajectory 1 as a 14th predictor alongside the 13 baseline logs.
2. **Experiment B (Physics-Informed with $DT_{\text{ratio}}$):** Adds four physically derived features (total 17 predictors): total vertical overburden stress ($S_v$), normal brine hydrostatic reference pressure ($P_h$), nominal hydrostatic effective stress ($\sigma_{v,\text{hydro}}$), and regional compaction slowness ratio ($DT / DT_{\text{nct}}$).
3. **Experiment B Sensitivity (Physics-Informed without $DT_{\text{ratio}}$):** Ablates the acoustic compaction ratio to evaluate the impact of removing the unphysical sonic normal compaction equation (total 16 predictors: 13 logs plus $S_v, P_h, \sigma_{v,\text{hydro}}$).
4. **Experiment C (Formation-Aware):** Adds four lithostratigraphic and diagenetic indicators (total 17 predictors): `IS_HUGIN`, `IS_SLEIPNER`, `IS_SKAGERRAK`, and `CALCITE_HARDGROUND`.

### Data Integrity Rules Across All Experiments
All experimental pipelines strictly enforce the following protocols:
- The contiguous 70:15:15 depth split is preserved identically (Training: 1,268 rows; Validation: 272 rows; Testing: 271 rows).
- All scalers, normalizers, and log10 transformations are fitted **exclusively on the 1,268 training rows**. Zero validation or testing rows are used in preprocessing.
- The 397 unsupported transition rows (MD 3,326.0 to 3,386.5 m) are excluded from training loss calculations and primary metrics.
- Physical stations **Test 20 and Test 21 are held out strictly for final evaluation**. They were never used to tune hyperparameters, adjust feature sets, or guide model selection.
- The original baseline models and output metrics remain completely unmodified.

---

### 9.1 Experiment A: Depth-Aware Models (+TVD_MSL)

#### Hypothesis and Architecture
Experiment A tests whether supplying a geometric depth coordinate (`TVD_MSL`, Trajectory 1) allows continuous regression models to learn the vertical pore-pressure trend and bridge the domain shift between the Sleipner and Skagerrak formations.

#### Key Findings
- **Apparent Success on Test Anchors:** The Multi-Layer Perceptron (MLP) achieved a combined MAE of **5.819 bar** across Tests 20 and 21 (predicting Test 20 at 364.404 bar and Test 21 at 376.154 bar), compared to 31.110 bar in the logs-only baseline.
- **Validation Penalty (Depth Overfitting):** However, on the unseen lower Sleipner validation checkpoint (Test 16 at 3,324.65 m MD), the depth-aware MLP overpredicted at **345.354 bar** (residual -3.641 bar), substantially worse than its exceptional baseline prediction (341.443 bar, residual +0.270 bar).
- **Decision Tree Plateauing:** Tree-based models (Random Forest, CatBoost, Gradient Boosting) split on depth thresholds. For any observation deeper than the maximum training depth ($\text{TVD} > 2,933.64$ m), tree models route samples to the deepest available terminal leaf. Consequently, Gradient Boosting outputs a constant flatline prediction of **341.422 bar** across the entire validation and testing interval, producing a constant residual plateau of +27.384 bar across Tests 20 and 21.

---

### 9.2 Experiment B: Physics-Informed Models (+Sv, +Ph, +sigma_v_hydro, +DT_nct_ratio)

#### Hypothesis and Architecture
Experiment B evaluates whether geomechanical and hydrodynamic boundary conditions derived from wireline density integration and regional compaction theory can guide models without direct depth memorization.

#### Key Findings
- **Parametric Scaling in MLR:** Total overburden stress ($S_v$) and normal hydrostatic pressure ($P_h$) provide a natural upward trend with burial depth. For Multiple Linear Regression (MLR), Tests 20 and 21 combined MAE dropped dramatically from **36.871 bar** in baseline to **13.242 bar** (predicting Test 20 at 352.355 bar and Test 21 at 358.686 bar).
- **MLP Physics Guidance:** The physics-informed MLP predicted Test 20 at **370.964 bar** (residual -2.258 bar) and Test 21 at **384.826 bar** (residual -16.008 bar, combined MAE = 9.133 bar), while predicting Test 16 at 345.964 bar.
- **Tree Model Response:** Tree ensembles captured non-linear effective stress interactions; Random Forest validation error on Test 16 improved to **1.049 bar** (predicted 340.665 bar). However, in the deep test interval, tree predictions still saturated around 340.5 to 341.0 bar.

---

### 9.3 Experiment B Sensitivity: Ablation of Sonic Compaction Ratio (exp_b_no_dtratio)

#### Rationale
As documented in Section 11, the regional sonic normal compaction trend equation ($DT_{\text{nct}} = 195 \exp(-0.00045 \cdot \text{TVD})$) produces unphysical values (~53 $\mu$s/ft at 2,900 m TVD, faster than pure quartz mineral grains). A controlled ablation was executed where $DT_{\text{ratio}}$ was removed, training all five models strictly on the 16 remaining predictors (13 logs plus $S_v, P_h, \sigma_{v,\text{hydro}}$).

#### Key Findings
- **MLR Invariance:** MLR performance remained virtually identical: Tests 20 and 21 combined MAE was **13.348 bar** (compared to 13.242 bar with $DT_{\text{ratio}}$), and Test 16 validation error was **1.812 bar** (predicted 339.901 bar). This demonstrates that MLR scaling is driven by $S_v$ and $P_h$, not by the corrupted sonic ratio.
- **Tree Model Stability:** Random Forest and CatBoost performed comparably or slightly better without $DT_{\text{ratio}}$: Random Forest Test 16 error was **1.232 bar** (predicted 340.481 bar), and CatBoost Test 20/21 MAE was **30.217 bar** (improving from 31.023 bar).
- **MLP Divergence from Multicollinearity:** Without $DT_{\text{ratio}}$ acting as an acoustic regularizer, the exact mathematical collinearity among the remaining physics features ($\sigma_{v,\text{hydro}} = S_v - P_h$) caused the unregularized neural network to extrapolate aggressively, predicting Test 20 at **397.826 bar** (residual -29.120 bar) and Test 21 at **425.052 bar** (residual -56.234 bar, MAE = 42.677 bar).

---

### 9.4 Experiment C: Formation-Aware Models (+Stratigraphy, +Hardgrounds)

#### Hypothesis and Architecture
Experiment C introduces categorical lithostratigraphic indicators (`IS_HUGIN`, `IS_SLEIPNER`, `IS_SKAGERRAK`) and calcite hardground cementing flags (`CALCITE_HARDGROUND`) to test whether stratigraphic context helps models segment overpressure regimes.

#### The Unseen Categorical Level Trap
In the training partition (MD $\le 3,320.8$ m), `IS_SKAGERRAK` is identically zero (0% representation). This creates a structural machine learning failure mode:
1. **Linear Models (MLR):** Because `IS_SKAGERRAK` has zero variance in training data, its regression coefficient is identically zero. In testing, it provides zero predictive power.
2. **Decision Trees (RF, CatBoost, GBR):** Tree split algorithms cannot place a split on a constant feature. When test samples arrive with `IS_SKAGERRAK == 1`, tree models route them down the default branch, predicting 332.0 to 334.2 bar (worse than baseline).
3. **Neural Networks (MLP):** Evaluating an untrained hidden neuron with an active categorical input produces severe numerical instability, causing the MLP to violently collapse to **301.884 bar** on Test 20 (residual +66.822 bar).
- **Methodological Conclusion:** Categorical stratigraphy cannot extrapolate pore pressure into an uncalibrated formation without prior regional pressure training samples from that formation.

---

## 10. Scientific Audit of Physics-Informed Features (Experiment B)

A rigorous audit was conducted to investigate whether the Experiment B predictors introduce circularity, target-construction leakage, or artificial reconstruction with the target-generation methodology.

### 10.1 Feature-by-Feature Formulation, Inputs, Assumptions, and Target Alignment

#### 1. Total Vertical Overburden Stress ($S_v$, bar)
- **Mathematical Equation:**
  $$S_v(z) = P_{\text{water}} + P_{\text{shallow}} + \int_{z_{\text{shallow}}}^z \rho_b(z') g dz'$$
- **Inputs:** Continuous wireline bulk density log (`RHOB`), Trajectory 1 TVD MSL ($z$), seawater depth 91.0 m ($1.025\text{ g/cm}^3$), shallow unlogged section reference density $1.90\text{ g/cm}^3$, $g = 9.80665\text{ m/s}^2$.
- **Assumptions & Uncertainty:** Vertical principal stress alignment; reference shallow density $1.90\text{ g/cm}^3$ for 91.0 to 262.0 m TVD MSL. Composite sensitivity across shallow density bounds ($1.80$ to $1.95\text{ g/cm}^3$) is $\pm 1.26$ bar ($\pm 0.21\%$).
- **Relationship to Target Labels:** Target pore pressures were generated from formation fluid gradients ($P = P_0 + \rho_{\text{fluid}} g \Delta z$) and discrete wireline pretest anchors. Overburden stress $S_v$ was **never used** to compute target pressures.
- **Circularity Assessment:** **Zero circularity.** $S_v$ represents the total lithostatic overburden boundary condition. It is a strictly independent physical rock-mechanics property.

#### 2. Normal Brine Hydrostatic Reference Pressure ($P_h$, bar)
- **Mathematical Equation:**
  $$P_h(z) = \rho_{\text{seawater}} g z_{\text{seabed}} + \rho_{\text{brine}} g (z - z_{\text{seabed}})$$
  Where $z_{\text{seabed}} = 91.0$ m, $\rho_{\text{seawater}} = 1.025\text{ g/cm}^3$ ($0.9145$ bar at seabed), and $\rho_{\text{brine}} = 1.030\text{ g/cm}^3$ ($0.10103\text{ bar/m}$).
- **Inputs:** Trajectory 1 TVD MSL, seabed depth 91.0 m, standard seawater and formation brine densities.
- **Assumptions & Uncertainty:** Continuous, normally pressured formation brine column open to the seabed datum; uniform water salinity. Uncertainty is $\pm 0.5$ bar across brine density range $1.025$ to $1.035\text{ g/cm}^3$.
- **Relationship to Target Labels & Target-Construction Slope Alignment:**
  - In target generation, Sleipner middle and lower sand targets were generated using a formation brine gradient of **$0.101\text{ bar/m}$** ($1.030\text{ g/cm}^3$), anchored to Test 14 and Test 16.
  - Notice: The brine fluid gradient used in target generation is mathematically identical to the depth derivative of the $P_h(z)$ predictor:
    $$\frac{dP_{\text{target}}}{dz} = \frac{dP_h}{dz} = 0.10103\text{ bar/m} \quad (\text{across all Sleipner brine intervals})$$
  - Overpressure is defined physically as $\Delta P(z) = P_{\text{target}}(z) - P_h(z)$. In the middle Sleipner reservoir sand, $\Delta P \approx +48.65$ bar (a constant offset across depth).
  - Consequently, providing $P_h(z)$ to a linear model gives it an exact carrier of the target slope. The linear model does not need to learn rock physics; it simply fits a slope coefficient of $w \approx 1.0$ and an intercept of approximately $+48.65$ bar.
  - **Circularity Assessment:** This is **not direct target leakage** (no measured target values or test numbers are provided to the model). However, it represents **target-construction slope alignment**: the input feature shares the exact mathematical slope ($0.101\text{ bar/m}$) chosen by the target-generation methodology for the training targets.

#### 3. Nominal Hydrostatic Effective Stress ($\sigma_{v,\text{hydro}}$, bar)
- **Mathematical Equation:**
  $$\sigma_{v,\text{hydro}}(z) = S_v(z) - P_h(z)$$
- **Inputs:** $S_v(z)$ and $P_h(z)$.
- **Assumptions & Uncertainty:** Nominal vertical effective stress under hypothetical normal pore-fluid pressure. Composite uncertainty is $\pm 1.4$ bar from $S_v$ and $P_h$.
- **Collinearity and Numerical Risk:** Because $\sigma_{v,\text{hydro}}$ is an exact linear combination of $S_v$ and $P_h$, the feature triplet $\{S_v, P_h, \sigma_{v,\text{hydro}}\}$ has a linear algebra rank of 2, creating exact multicollinearity. While regularized linear regression (MLR) handles this stably via singular value decomposition, unregularized neural networks (MLP) suffer severe weight divergence when extrapolated outside training bounds, as evidenced by the MLP's 425 bar prediction in the ablation experiment.

#### 4. Sonic Compaction Slowness Ratio ($DT / DT_{\text{nct}}$)
- **Mathematical Equation:**
  $$DT_{\text{nct}}(z) = DT_0 \exp(-c \cdot z), \quad DT_{\text{ratio}} = \frac{DT}{DT_{\text{nct}}(z)}$$
- **Relationship to Target Labels:** Target generation never used sonic logs or compaction equations.
- **Circularity Assessment:** **Zero circularity with target labels.** However, as detailed in Section 11, the equation itself is physically defective.

### 10.2 Summary: Distinguishing Physical Correlation from Target-Construction Alignment

| Feature Name | Equation Inputs | Target Generation Used? | Direct Leakage? | Circularity / Alignment Assessment | Recommendation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `SV_OVERBURDEN_BAR` | RHOB, TVD_MSL, 91m seabed | No | No | **Zero Circularity.** Pure physical lithostatic boundary condition. | Retain as sound physical feature. |
| `PH_HYDROSTATIC_BAR` | TVD_MSL, 1.030 g/cm3 brine | Yes (Brine gradient) | No | **Target-Construction Slope Alignment.** Shares exact target slope ($0.101$ bar/m) in Sleipner brine. | Retain only with explicit disclosure of slope coupling. |
| `SIGMA_V_HYDRO_BAR` | $S_v - P_h$ | No | No | **Collinear with $S_v$ and $P_h$.** Destabilizes neural networks under extrapolation. | Redundant if $S_v$ and $P_h$ are both present. |
| `DT_NCT_RATIO` | DT, TVD_MSL, regional constants | No | No | **Zero Target Leakage, but Physically Defective Equation.** Predicts sub-quartz transit time (~53 $\mu$s/ft). | **Reject.** Must be removed from physical feature set. |

---

## 11. Sonic NCT Equation Investigation and Matrix Bounds Analysis

The equation currently used in Experiment B is:
$$DT_{\text{nct}}(z) = 195 \exp(-0.00045 \cdot z)$$
Where $z$ is True Vertical Depth Subsea (`TVD_MSL`, m MSL). At $z = 2,900$ m TVD, this equation outputs:
$$DT_{\text{nct}}(2900) = 195 \exp(-0.00045 \times 2900) = 195 \exp(-1.305) = 195 \times 0.27117 = \mathbf{52.88\ \mu\text{s/ft}}$$

This produces a fundamental scientific discrepancy: an earlier investigation of mudstone intervals in Well 15/9-F-12 established normal shale compaction transit times of approximately **111 to 114 $\mu$s/ft** at 2,500 to 2,600 m depth (`tables/shale_sonic_depth_bins.csv`). A rigorous investigation identified five fatal flaws in the current equation:

### 1. Imperial-to-Metric Unit Mismatch
In classical petroleum literature (e.g., Eaton 1975, Magara 1978, Chapman 1983):
- A surface mudline transit time $DT_0 \approx 195$ to $200\ \mu$s/ft is standard for unconsolidated seafloor muds ($V_p \approx 1,560$ m/s).
- However, the exponential compaction coefficient $c \approx 0.00025$ to $0.00045$ is specified per **foot** of depth ($c \approx 0.00045\text{ ft}^{-1}$), not per meter!
- Because $1\text{ meter} \approx 3.28084\text{ feet}$, applying $c = 0.00045\text{ m}^{-1}$ directly to depth in meters effectively triples the compaction rate. The depth required to compact to a given transit time is compressed by a factor of 3.28.

### 2. Violation of Mineral Matrix Slowness Bounds
In physical acoustics, the acoustic transit time of a porous rock cannot be lower than the transit time of its solid mineral matrix ($DT_{\text{matrix}}$):
- **Pure Quartz Matrix:** $DT_{\text{matrix}} \approx 55.5\ \mu$s/ft ($V_{\text{matrix}} \approx 5,486$ m/s or 18,000 ft/s).
- **Pure Calcite Matrix:** $DT_{\text{matrix}} \approx 47.6\ \mu$s/ft ($V_{\text{matrix}} \approx 6,400$ m/s).
- **Pure Shale / Clay Matrix:** $DT_{\text{matrix}} \approx 67.0$ to $72.0\ \mu$s/ft ($V_{\text{matrix}} \approx 4,200$ to $4,500$ m/s).
- A simple exponential decay $DT_0 \exp(-c z)$ without a matrix asymptote decays toward zero as depth increases. At 2,900 m TVD, predicting $DT_{\text{nct}} = 52.88\ \mu$s/ft asserts that normally compacting shale travels faster than solid quartz rock. Standard compaction trends must incorporate an asymptotic matrix floor:
  $$DT_{\text{nct}}(z) = DT_{\text{matrix}} + (DT_0 - DT_{\text{matrix}}) \exp(-c \cdot z)$$

### 3. Lithological and Diagenetic Incompatibility
- Normal compaction trend analysis is derived exclusively for thick, mechanically compacting Cenozoic mudstones where disequilibrium compaction is the primary overpressure mechanism.
- The studied reservoir interval in Well 15/9-F-12 (MD 3,127.7 to 3,403.5 m) consists of Middle-to-Late Jurassic sandstones (Hugin, Sleipner, Skagerrak) subjected to severe quartz overgrowth cementation, grain dissolution, and calcite hardgrounds.
- In cemented reservoir sandstones, sonic velocity is controlled by grain frame stiffness, sorting, and pore-fluid compressibility (Gassmann equation), not mechanical compaction disequilibrium. Applying a shale NCT to reservoir sandstones is methodologically invalid.

### 4. Conflict with Measured Borehole Logs in 15/9-F-12
- In the overlying Tertiary and Cretaceous shale sequence (1,500 to 3,000 m MD), empirical regression across Gamma Ray thresholds ($\text{GR} \ge 60$ to $90$ API, documented in `tables/nct_trend_parameters.csv`) yielded:
  - Surface transit time: $DT_0 = 173.7$ to $189.6\ \mu$s/ft
  - Compaction coefficient: $c = 0.000151$ to $0.000204\text{ m}^{-1}$
  - Projected normal shale transit time at 3,128 m MD: **$100.2$ to $108.3\ \mu$s/ft**
- In the reservoir interval, measured DT ranges from 56.8 to 100.4 $\mu$s/ft (median $83.9\ \mu$s/ft). Dividing measured DT (~84 $\mu$s/ft) by an unphysical $DT_{\text{nct}} \approx 53\ \mu$s/ft produces $DT_{\text{ratio}} \approx 1.58$.
- In pore-pressure interpretation, a ratio of 1.58 indicates severe undercompaction overpressure, whereas the rocks are actually normally pressured to moderately overpressured, heavily cemented sandstones.

### 5. Sensitivity Ablation Results
Ablating $DT_{\text{ratio}}$ in `exp_b_no_dtratio` confirms that removing this feature preserves MLR physical scaling (Tests 20 and 21 MAE = 13.348 bar vs 13.242 bar) and improves tree model stability, confirming that $DT_{\text{ratio}}$ provided no physical predictive value and should be permanently removed.

---

## 12. Robustness Assessment of Depth-Aware MLP (Experiment A)

### 12.1 The Mathematical Extrapolation Paradox
Experiment A revealed a striking divergence in the Multi-Layer Perceptron (MLP):
- In the testing partition (Skagerrak Formation, Tests 20 & 21), adding `TVD_MSL` reduced combined MAE from **31.110 bar** to **5.819 bar**.
- Simultaneously, on the unseen lower Sleipner validation checkpoint (**Test 16** at 3,324.65 m MD, 2,935.97 m TVD MSL), the depth-aware MLP overpredicted at **345.354 bar** (residual -3.641 bar), completely degrading from its near-perfect baseline prediction (**341.443 bar**, residual **+0.270 bar**).

To determine whether the depth-aware MLP learned a true physical overpressure trend or merely an unconstrained mathematical slope, a systematic sensitivity analysis was executed across the borehole.

### 12.2 Empirical Depth Sensitivity Sweep Analysis
Holding all 13 petrophysical logs fixed at their median reservoir sandstone values, `TVD_MSL` was swept continuously from 2,800 m to 3,050 m in 1.0 m increments ($N=251$ points). At each depth, the predicted pressure $\hat{P}(\text{TVD})$ and its local empirical depth gradient $\frac{d\hat{P}}{d\text{TVD}}$ (bar/m) were evaluated. The full profile is preserved in `tables/experiments/mlp_depth_sensitivity_profile.csv`.

#### Table 4: Depth Sensitivity Profile of Experiment A MLP Across Borehole Regimes

| TVD MSL (m) | Stratigraphic / Partition Regime | Predicted $P_p$ (bar) | Local Gradient $\frac{d\hat{P}}{d\text{TVD}}$ (bar/m) | Equivalent Fluid Density ($\text{g/cm}^3$) | Geomechanical Assessment |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **2,800.0** | Pre-Training Shallow | 318.40 | -0.278 | -2.83 | Negative unconstrained boundary artifact |
| **2,825.0** | Training: Upper Hugin Oil Sand | 315.87 | +0.063 | 0.64 | Matches Hugin oil gradient ($0.070$ bar/m) |
| **2,850.0** | Training: Main Hugin Reservoir | 317.44 | +0.065 | 0.66 | Stable hydrocarbon gradient |
| **2,875.0** | Training: Lower Hugin Sand | 320.52 | +0.160 | 1.64 | Steepening toward water contact |
| **2,900.0** | Training: Middle Sleipner Sand | 324.63 | +0.173 | 1.77 | Transition into overpressured brine |
| **2,925.0** | Training: Base Training Partition | 330.20 | +0.333 | 3.39 | **Accelerating convex curvature** |
| **2,950.0** | Validation: Lower Sleipner | 338.60 | +0.341 | 3.48 | **Degrades Test 16 (+3.64 bar overshoot)** |
| **2,975.0** | Testing: Skagerrak Compartment | 350.64 | +0.544 | 5.55 | **2.5x Lithostatic Overburden Gradient** |
| **3,000.0** | Extrapolated Deep Sub-Skagerrak | 365.50 | +0.686 | 7.00 | Runaway mathematical slope |
| **3,025.0** | Extrapolated Deep Sub-Skagerrak | 383.39 | +0.753 | 7.68 | Runaway mathematical slope |
| **3,050.0** | Extrapolated Deep Sub-Skagerrak | 403.86 | +0.843 | 8.59 | Super-lithostatic runaway divergence |

### 12.3 Mechanistic Explanation of Extrapolation Failure
Inspection of Table 4 provides conclusive physical evidence:
1. **The Training Transition Slope Artifact:**
   Inside the training partition, pore pressure increases from 320.02 bar (Hugin oil at 2,819.4 m TVD) to 335.61 bar (middle Sleipner brine at 2,933.6 m TVD). Because the training set spans an 8 to 9 bar hydrodynamic step between the Hugin oil reservoir and the Sleipner brine aquifer, the apparent depth slope steepens from $0.063\text{ bar/m}$ to $0.333\text{ bar/m}$ near the base of the training interval.
2. **Convex ReLU Extrapolation:**
   A feedforward neural network with Rectified Linear Unit (ReLU) activations fits piece-wise linear affine hyperplanes. Once all hidden neurons activated by positive depth inputs remain turned on outside the training range, their positive weights compound monotonically. The MLP does not predict a constant compartment pressure; it projects an accelerating upward slope.
3. **The Test 16 Validation Penalty:**
   At Test 16 (TVD 2,935.97 m), the network's fitted depth gradient is already $0.341\text{ bar/m}$ ($3.48\text{ g/cm}^3$ equivalent fluid density). This runaway slope pushes the prediction to 345.35 bar, overshooting the physical measurement (341.713 bar) by 3.64 bar.
4. **The False Coincidence of Tests 20 & 21:**
   Tests 20 and 21 lie approximately 40 to 50 meters deeper than the training interval. Projecting a gradient of $0.54$ to $0.61\text{ bar/m}$ over 40 meters adds approximately 25 bar to the training base, projecting into the 365 to 376 bar range purely by geometric accident.
5. **The Unphysical Inter-Station Gradient:**
   The ultimate proof of failure lies in the predicted difference between Test 20 and Test 21:
   - Test 20 is at TVD 2,974.77 m; Test 21 is at TVD 2,983.34 m ($\Delta \text{TVD} = 8.57$ m).
   - The depth-aware MLP predicts Test 20 at **364.404 bar** and Test 21 at **376.154 bar**, a predicted difference of **11.750 bar** over 8.57 meters!
   - This corresponds to a vertical pressure gradient of:
     $$\frac{\Delta \hat{P}}{\Delta \text{TVD}} = \frac{11.750\text{ bar}}{8.57\text{ m}} = \mathbf{1.371\text{ bar/m}} \quad (\text{Equivalent fluid density } \approx \mathbf{13.98\text{ g/cm}^3})$$
   - An equivalent fluid density of $14.0\text{ g/cm}^3$ is denser than liquid mercury ($13.55\text{ g/cm}^3$) and over six times the total lithostatic overburden stress ($2.15\text{ g/cm}^3$).
   - Yet in reality, the physical measurements at Tests 20 and 21 are **368.706 bar** and **368.818 bar**, a difference of only **0.112 bar**!
- **Conclusion:** The depth-aware MLP did **not** learn the Skagerrak overpressure compartment. It fitted an unconstrained mathematical affine slope that coincidentally intersected the Skagerrak pressure range at 2,975 m before violently diverging. Experiment A must be **rejected** for operational deployment.

---

## 13. Comprehensive Individual Physical Station Error Matrix and Benchmark Summary

To avoid the masking effects of aggregate MAE and RMSE metrics, model performance is reported on an individual station-by-station basis for every candidate model across all five experimental configurations.

### 13.1 Key Physical Evaluation Stations
- **Test 16 (Validation Checkpoint):** MD 3,324.65 m, TVD MSL 2,935.92 m, Lower Sleipner Formation. Physical measured buildup = **341.713 bar**. Evaluates generalization across the intermediate hydrodynamic step.
- **Test 20 (Independent Test Anchor):** MD 3,386.99 m, TVD MSL 2,974.77 m, Skagerrak Formation. Physical measured buildup = **368.706 bar**. First independent anchor in the deep overpressured compartment.
- **Test 21 (Independent Test Anchor):** MD 3,403.57 m, TVD MSL 2,983.34 m, Skagerrak Formation. Physical measured buildup = **368.818 bar**. Second independent anchor in the deep compartment.
- **Test 19 (Excluded Conflicting Memory Pretest):** MD 3,403.53 m, TVD MSL 2,983.31 m, Skagerrak Formation. Tool telemetry failed; memory gauge recovered **357.714 bar**. Excluded from primary calibration anchors; retained exclusively as an independent conflicting reference.

### Table 5: Individual Physical Station Error Matrix Across All Models and Experiments

| Experiment Configuration | Model Family | Test 16 Pred (bar) | Test 16 Res (bar) | Test 16 AbsErr (bar) | Test 20 Pred (bar) | Test 20 Res (bar) | Test 20 AbsErr (bar) | Test 21 Pred (bar) | Test 21 Res (bar) | Test 21 AbsErr (bar) | Tests 20 & 21 MAE (bar) | Test 19 Pred (bar) | Test 19 Diff (bar) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logs-Only)** | MLR | 335.822 | +5.892 | 5.892 | 333.921 | +34.785 | 34.785 | 329.860 | +38.958 | 38.958 | 36.871 | 329.860 | +27.854 |
| **Baseline (Logs-Only)** | Random Forest | 334.801 | +6.912 | 6.912 | 335.295 | +33.411 | 33.411 | 338.038 | +30.780 | 30.780 | 32.095 | 338.038 | +19.676 |
| **Baseline (Logs-Only)** | CatBoost | 335.233 | +6.480 | 6.480 | 336.867 | +31.839 | 31.839 | 338.341 | +30.477 | 30.477 | 31.158 | 338.341 | +19.373 |
| **Baseline (Logs-Only)** | Gradient Boosting | 335.458 | +6.255 | 6.255 | 336.586 | +32.121 | 32.121 | 340.123 | +28.695 | 28.695 | 30.408 | 340.123 | +17.591 |
| **Baseline (Logs-Only)** | MLP | **341.443** | **+0.270** | **0.270** | 335.323 | +33.383 | 33.383 | 339.982 | +28.836 | 28.836 | 31.110 | 339.982 | +17.732 |
| **Exp A (+TVD_MSL)** | MLR | 337.537 | +4.176 | 4.176 | 339.163 | +29.543 | 29.543 | 340.246 | +28.572 | 28.572 | 29.057 | 340.246 | +17.468 |
| **Exp A (+TVD_MSL)** | Random Forest | 338.346 | +3.368 | 3.368 | 338.512 | +30.194 | 30.194 | 340.358 | +28.460 | 28.460 | 29.327 | 340.358 | +17.356 |
| **Exp A (+TVD_MSL)** | CatBoost | 337.855 | +3.858 | 3.858 | 337.791 | +30.915 | 30.915 | 339.196 | +29.622 | 29.622 | 30.269 | 339.196 | +18.518 |
| **Exp A (+TVD_MSL)** | Gradient Boosting | 341.422 | +0.291 | 0.291 | 341.401 | +27.305 | 27.305 | 341.355 | +27.463 | 27.463 | 27.384 | 341.355 | +16.359 |
| **Exp A (+TVD_MSL)** | MLP | 345.354 | -3.641 | 3.641 | **364.404** | **+4.302** | **4.302** | **376.154** | **-7.336** | **7.336** | **5.819** | 376.154 | -18.440 |
| **Exp B (+Physics)** | MLR | 339.484 | +2.229 | 2.229 | 352.355 | +16.351 | 16.351 | 358.686 | +10.132 | 10.132 | 13.242 | 358.686 | -0.972 |
| **Exp B (+Physics)** | Random Forest | 340.665 | +1.049 | 1.049 | 340.536 | +28.170 | 28.170 | 340.961 | +27.857 | 27.857 | 28.014 | 340.961 | +16.753 |
| **Exp B (+Physics)** | CatBoost | 337.297 | +4.416 | 4.416 | 336.822 | +31.884 | 31.884 | 338.657 | +30.161 | 30.161 | 31.023 | 338.657 | +19.057 |
| **Exp B (+Physics)** | Gradient Boosting | 341.461 | +0.252 | 0.252 | 341.439 | +27.267 | 27.267 | 341.395 | +27.423 | 27.423 | 27.345 | 341.395 | +16.319 |
| **Exp B (+Physics)** | MLP | 345.964 | -4.251 | 4.251 | 370.964 | -2.258 | 2.258 | 384.826 | -16.008 | 16.008 | 9.133 | 384.826 | -27.112 |
| **Exp B No DT Ratio**| MLR | 339.901 | +1.812 | 1.812 | 353.658 | +15.048 | 15.048 | 357.170 | +11.648 | 11.648 | 13.348 | 357.170 | +0.544 |
| **Exp B No DT Ratio**| Random Forest | 340.481 | +1.232 | 1.232 | 340.193 | +28.513 | 28.513 | 341.183 | +27.635 | 27.635 | 28.074 | 341.183 | +16.531 |
| **Exp B No DT Ratio**| CatBoost | 337.644 | +4.069 | 4.069 | 338.233 | +30.473 | 30.473 | 338.857 | +29.961 | 29.961 | 30.217 | 338.857 | +18.857 |
| **Exp B No DT Ratio**| Gradient Boosting | 341.447 | +0.266 | 0.266 | 341.436 | +27.270 | 27.270 | 341.387 | +27.431 | 27.431 | 27.351 | 341.387 | +16.327 |
| **Exp B No DT Ratio**| MLP | 352.599 | -10.886 | 10.886 | 397.826 | -29.120 | 29.120 | 425.052 | -56.234 | 56.234 | 42.677 | 425.052 | -67.338 |
| **Exp C (+Formation)**| MLR | 335.502 | +6.211 | 6.211 | 330.613 | +38.093 | 38.093 | 327.454 | +41.364 | 41.364 | 39.729 | 327.454 | +30.260 |
| **Exp C (+Formation)**| Random Forest | 335.051 | +6.662 | 6.662 | 332.036 | +36.670 | 36.670 | 334.184 | +34.634 | 34.634 | 35.652 | 334.184 | +23.530 |
| **Exp C (+Formation)**| CatBoost | 334.941 | +6.772 | 6.772 | 332.277 | +36.429 | 36.429 | 332.537 | +36.282 | 36.282 | 36.355 | 332.537 | +25.178 |
| **Exp C (+Formation)**| Gradient Boosting | 335.352 | +6.361 | 6.361 | 332.839 | +35.867 | 35.867 | 333.942 | +34.876 | 34.876 | 35.372 | 333.942 | +23.772 |
| **Exp C (+Formation)**| MLP | 338.514 | +3.199 | 3.199 | 301.884 | +66.822 | 66.822 | 319.101 | +49.717 | 49.717 | 58.270 | 319.101 | +38.613 |

*(Note: Residual = Actual minus Predicted; Positive residual indicates underprediction; Negative residual indicates overprediction. For Test 19, Diff = 357.714 minus Predicted).*

---

### 13.2 Partition Performance Breakdown

### Table 6: Global and Partition Metrics Across Baseline and All Experiments (Valid Targets Only)

| Experiment Configuration | Model Family | Training RMSE (bar) | Training $R^2$ | Validation MAE (bar) | Validation RMSE (bar) | Test Anchors MAE (bar) | Test Prov. MAE (bar) | Full Valid RMSE (bar) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logs-Only)** | MLR | 1.9334 | 0.8888 | 5.9796 | 5.9925 | 36.8713 | 36.1347 | 10.3934 |
| **Baseline (Logs-Only)** | Random Forest | 0.3319 | 0.9967 | 6.5875 | 6.6139 | 32.0953 | 33.1743 | 9.3984 |
| **Baseline (Logs-Only)** | CatBoost | **0.2023** | **0.9988** | 6.3066 | 6.3345 | 31.1577 | 33.5800 | 9.4958 |
| **Baseline (Logs-Only)** | Gradient Boosting | 0.2168 | 0.9986 | 6.3357 | 6.3508 | 30.4076 | 33.0321 | 9.3565 |
| **Baseline (Logs-Only)** | MLP | 1.4951 | 0.9335 | **2.7071** | **3.5882** | 31.1097 | **31.2473** | **8.9742** |
| **Exp A (+TVD_MSL)** | MLR | 1.5471 | 0.9288 | 4.5068 | 4.5376 | 29.0574 | 29.2182 | 8.3582 |
| **Exp A (+TVD_MSL)** | Random Forest | 0.1652 | 0.9992 | 3.0064 | 3.1235 | 29.3270 | 29.7540 | 8.3697 |
| **Exp A (+TVD_MSL)** | CatBoost | 0.0886 | 0.9998 | 3.7414 | 3.8055 | 30.2685 | 29.7323 | 8.3842 |
| **Exp A (+TVD_MSL)** | Gradient Boosting | **0.0121** | **1.0000** | **0.2320** | **0.2842** | 27.3839 | 27.3931 | 7.7142 |
| **Exp A (+TVD_MSL)** | MLP | 0.8841 | 0.9768 | 2.8489 | 3.2384 | **5.8189** | **3.1918** | **1.9442** |
| **Exp B (+Physics)** | MLR | 1.4429 | 0.9381 | 2.3998 | 2.5358 | 13.2415 | 14.0929 | 4.2587 |
| **Exp B (+Physics)** | Random Forest | 0.0882 | 0.9998 | 0.8708 | 1.0543 | 28.0134 | 27.8732 | 7.8483 |
| **Exp B (+Physics)** | CatBoost | 0.0632 | 0.9999 | 4.0073 | 4.0984 | 31.0225 | 30.4778 | 8.5772 |
| **Exp B (+Physics)** | Gradient Boosting | **0.0125** | **1.0000** | **0.1841** | **0.2312** | 27.3449 | 27.3503 | 7.7018 |
| **Exp B (+Physics)** | MLP | 1.1245 | 0.9624 | 2.7580 | 3.3211 | **9.1327** | **4.9532** | **2.2854** |
| **Exp B No DT Ratio**| MLR | 1.2835 | 0.9510 | 2.1554 | 2.1987 | 13.3481 | 13.5439 | 4.0254 |
| **Exp B No DT Ratio**| Random Forest | 0.0701 | 0.9999 | 1.1455 | 1.1970 | 28.0740 | 27.9951 | 7.8822 |
| **Exp B No DT Ratio**| CatBoost | 0.0607 | 0.9999 | 3.5860 | 3.7392 | 30.2171 | 29.7668 | 8.4033 |
| **Exp B No DT Ratio**| Gradient Boosting | **0.0073** | **1.0000** | **0.2065** | **0.2311** | 27.3508 | 27.3639 | 7.7013 |
| **Exp B No DT Ratio**| MLP | 1.3292 | 0.9474 | 8.0625 | 8.8670 | 42.6770 | 39.1247 | 11.3453 |
| **Exp C (+Formation)**| MLR | 1.5421 | 0.9292 | 6.2460 | 6.2624 | 39.7288 | 39.0200 | 11.1345 |
| **Exp C (+Formation)**| Random Forest | 0.1852 | 0.9990 | 6.4676 | 6.4889 | 35.6522 | 36.6274 | 10.3245 |
| **Exp C (+Formation)**| CatBoost | 0.1741 | 0.9991 | 6.5850 | 6.6120 | 36.3553 | 38.0059 | 10.7023 |
| **Exp C (+Formation)**| Gradient Boosting | **0.1702** | **0.9992** | 6.3148 | 6.3402 | 35.3717 | 37.2200 | 10.4851 |
| **Exp C (+Formation)**| MLP | 0.7241 | 0.9844 | **4.9438** | **5.0889** | 58.2695 | 57.5979 | 16.1452 |

---

### 13.3 Performance Breakdown by Target Provenance

All evaluations must distinguish between measured ground truth and physics-based or provisional targets:
1. **Measured Pretest Anchors (`measured_anchor`, $n=17$):** True physical pressure measurements from Schlumberger FPWD pretests (14 in Training, 1 in Validation, 2 in Testing).
2. **Physics-Derived Continuous Targets (`physics_derived`, $n=1,287$):** Hydrodynamically interpolated target labels computed from verified oil and brine density gradients (1,254 in Training, 33 in Validation).
3. **Provisional Assumptions (`provisional_assumption`, $n=110$):** Flat 368.762 bar values assigned to the Skagerrak interval based on the assumption of a single hydraulic compartment. **These must never be presented as independent physical validation.** Deviation from these targets reflects model divergence from an unverified modeling assumption, not from physical ground truth.

### Table 7: Comprehensive Provenance Breakdown Across Key Model Configurations

| Configuration | Model | Provenance Category | Partition Subset | N Obs | MAE (bar) | RMSE (bar) | Mean Residual (bar) | MaxAE (bar) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | CatBoost | `measured_anchor` | Training (Tests 1-12, 14, 15) | 14 | **0.1452** | **0.2683** | -0.0996 | **0.9418** |
| **Baseline** | CatBoost | `measured_anchor` | Validation (Test 16) | 1 | 6.4804 | 6.4804 | +6.4804 | 6.4804 |
| **Baseline** | CatBoost | `measured_anchor` | Testing (Tests 20 & 21) | 2 | 31.1577 | 31.1652 | +31.1577 | 31.8388 |
| **Baseline** | CatBoost | `physics_derived` | Training (Hugin/Sleipner) | 1,254 | **0.1395** | **0.2014** | +0.0015 | **1.3851** |
| **Baseline** | CatBoost | `physics_derived` | Validation (Lower Sleipner) | 33 | 6.3013 | 6.3300 | +6.3013 | 7.6325 |
| **Baseline** | CatBoost | `provisional_assumption`| Testing (Skagerrak Assumed) | 110 | 33.5800 | 33.5942 | +33.5800 | 35.4199 |
| **Baseline** | MLP | `measured_anchor` | Training (Tests 1-12, 14, 15) | 14 | 0.8081 | 1.0444 | -0.6062 | 2.2567 |
| **Baseline** | MLP | `measured_anchor` | Validation (Test 16) | 1 | **0.2700** | **0.2700** | **+0.2700** | **0.2700** |
| **Baseline** | MLP | `measured_anchor` | Testing (Tests 20 & 21) | 2 | 31.1097 | 31.1926 | +31.1097 | 33.3831 |
| **Baseline** | MLP | `physics_derived` | Training (Hugin/Sleipner) | 1,254 | 0.9956 | 1.4993 | -0.0285 | 11.1011 |
| **Baseline** | MLP | `physics_derived` | Validation (Lower Sleipner) | 33 | **2.7809** | **3.6419** | **+2.3282** | 8.2514 |
| **Baseline** | MLP | `provisional_assumption`| Testing (Skagerrak Assumed) | 110 | **31.2473** | **31.4296** | **+31.2473** | 36.8874 |
| **Exp B (No DT)** | MLR | `measured_anchor` | Training (Tests 1-12, 14, 15) | 14 | 0.8412 | 1.0124 | -0.2145 | 1.6241 |
| **Exp B (No DT)** | MLR | `measured_anchor` | Validation (Test 16) | 1 | 1.8121 | 1.8121 | +1.8121 | 1.8121 |
| **Exp B (No DT)** | MLR | `measured_anchor` | Testing (Tests 20 & 21) | 2 | **13.3481** | **13.4559** | +13.3481 | 15.0480 |
| **Exp B (No DT)** | MLR | `physics_derived` | Training (Hugin/Sleipner) | 1,254 | 0.9761 | 1.2862 | +0.0024 | 3.5412 |
| **Exp B (No DT)** | MLR | `physics_derived` | Validation (Lower Sleipner) | 33 | 2.1658 | 2.2091 | +2.1658 | 2.8415 |
| **Exp B (No DT)** | MLR | `provisional_assumption`| Testing (Skagerrak Assumed) | 110 | **13.5439** | **13.5839** | +13.5439 | 15.6214 |

---

## 14. Methodological Recommendations on Retaining or Rejecting Experiments

Based on rigorous data integrity audits, mathematical extrapolation mechanics, and physical validation against discrete pretest stations, the following recommendations are established:

### 1. Experiment A (Depth-Aware: +TVD_MSL): REJECT FOR OPERATIONAL DEPLOYMENT
- **Recommendation:** **Reject** for general operational deployment and reservoir modeling. Retain in the thesis exclusively as a **cautionary diagnostic study** on depth-memorization failure modes.
- **Scientific Justification:** While the depth-aware MLP achieved an apparently low MAE on Tests 20 & 21 (5.82 bar), our depth-derivative audit proves this is an unconstrained mathematical artifact. The network learned an accelerating, convex upward depth derivative ($0.54$ to $0.84	ext{ bar/m}$, 2.5x to 4x lithostatic overburden gradient). This caused it to degrade validation Test 16 by 3.64 bar and predict an impossible 11.75 bar difference over 8.57 m TVD between Tests 20 and 21 ($14.0	ext{ g/cm}^3$ fluid density). Depth-aware models fail to transfer across structural faults or into offset wells where overpressure compartments do not align with geometric subsea depth.

### 2. Experiment B (Physics-Informed with $DT_{	ext{ratio}}$): REJECT
- **Recommendation:** **Reject.**
- **Scientific Justification:** The acoustic compaction ratio ($DT / DT_{	ext{nct}}$) is fatally compromised by an imperial-to-metric unit mismatch in the regional NCT equation ($0.00045	ext{ ft}^{-1}$ applied to meters), predicting an unphysical sub-quartz transit time (~53 $\mu$s/ft) at 2,900 m TVD. Furthermore, mechanical compaction trends cannot be applied to diagenetically cemented Jurassic reservoir sandstones. Incorporating this corrupted feature introduces artificial acoustic bias.

### 3. Experiment B Sensitivity (Physics-Informed without $DT_{	ext{ratio}}$): PARTIALLY RETAIN WITH DISCLOSURE
- **Recommendation:** **Partially retain** for linear regression models with full documentation of target-construction slope alignment; **reject** for unregularized neural networks.
- **Scientific Justification:** For linear models (MLR), total overburden stress ($S_v$) and hydrostatic pressure ($P_h$) provide sound geomechanical scaling, dropping Test 20/21 MAE to 13.35 bar without geometric depth coordinates. However, documentation must explicitly disclose that $P_h(z)$ shares the exact fluid gradient ($0.101	ext{ bar/m}$) used to construct Sleipner targets. For neural networks (MLP), exact collinearity ($\sigma_{v,	ext{hydro}} = S_v - P_h$) destabilizes unregularized weights, causing severe divergence (predicting 425 bar at Test 21).

### 4. Experiment C (Formation-Aware: +Stratigraphy, +Hardgrounds): REJECT
- **Recommendation:** **Reject.**
- **Scientific Justification:** Introducing categorical formation indicators where an overpressured compartment (`IS_SKAGERRAK`) has 0% representation in the training set creates an unseen categorical level trap. Tree models default to training averages, and neural networks suffer extreme numerical collapse (MLP predicted 301.88 bar on Test 20). Categorical formation flags cannot extrapolate without prior calibration samples.

### 5. Primary Modeling Baseline (Logs-Only: 13 Features): RETAIN AS PRIMARY BASELINE
- **Recommendation:** **Retain** as the primary, physically honest baseline.
- **Scientific Justification:** 
  - **In-Distribution Reservoir Characterization:** CatBoost achieves unmatched stability and precision within calibrated sequences (14-anchor calibration RMSE = 0.268 bar; training RMSE = 0.202 bar).
  - **Out-of-Interval Validation Generalization:** The Multi-Layer Perceptron (MLP) achieves superior generalization across uncalibrated stratigraphy without depth coordinates, predicting Test 16 with an absolute error of only **0.270 bar**.
  - All models plateau in the deep Skagerrak because petrophysical logging tools in tight sandstone do not register the 30 bar hydraulic jump across the unconformity. Acknowledging this physical limitation is scientifically superior to masking it with depth memorization or unphysical features.

---

## 15. Complete Deliverable Manifest

All baseline and experimental models, metrics, tables, and figures are dual-saved and verified:

- **Serialized Model Artifacts (`outputs/models/` and `outputs/experiments/models/`):**
  - Baseline: `mlr_model.joblib`, `random_forest_model.joblib`, `catboost_model.joblib`, `gradient_boosting_model.joblib`, `mlp_model.joblib`, `log_standard_scaler.joblib`
  - Experiments: `{exp_a, exp_b, exp_b_no_dtratio, exp_c}_{mlr, random_forest, catboost, gradient_boosting, mlp}_model.joblib`
  - Scalers: `{exp_a, exp_b, exp_b_no_dtratio, exp_c}_scaler.joblib`
- **Dual-Saved Metrics Tables (Mirrored in `outputs/experiments/metrics/` and `tables/experiments/`):**
  - Predictions: `exp_a_predictions.{csv,json}`, `exp_b_predictions.{csv,json}`, `exp_b_no_dtratio_predictions.{csv,json}`, `exp_c_predictions.{csv,json}`
  - Summary Metrics: `experiment_comparison_summary.{csv,json}`
  - Station Benchmarks: `experiment_station_benchmarks.{csv,json}`
  - Provenance Metrics: `experiment_provenance_metrics.{csv,json}`
  - Individual Station Errors: `individual_physical_station_errors.{csv,json}` (long format, $N=100$)
  - Individual Station Benchmark Matrix: `individual_station_benchmark_matrix.{csv,json}` (wide format, $N=25$)
  - Depth Sensitivity Profile: `mlp_depth_sensitivity_profile.{csv,json}` ($N=251$)
  - Feature Metadata: `experiment_b_physics_features_metadata.json`, `experiment_b_no_dtratio_physics_features_metadata.json`, `experiment_c_formation_features_metadata.json`
- **Diagnostic Figures (`outputs/figures/` and `outputs/experiments/figures/`):**
  - `actual_vs_predicted.png`, `residuals_vs_depth_and_dist.png`, `borehole_pressure_profiles.png`, `feature_importance_comparison.png`, `learning_curves.png`
  - `outputs/experiments/figures/all_experiments_borehole_comparison.png`
- **Automated Verification:**
  - `tests/test_data_validation.py` (7 tests pass)
  - `tests/test_model_training.py` (7 tests pass)
  - `tests/test_physics_models.py` (11 tests pass)
  - `tests/test_target_generation.py` (8 tests pass)
  - `tests/test_experiments.py` (7 tests pass)
  - **Total Test Suite:** 40 tests, 100% pass rate.
