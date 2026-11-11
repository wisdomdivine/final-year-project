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

To systematically investigate whether non-petrophysical features can bridge the observed domain shifts without compromising geological integrity, three controlled experiments were conducted. All three experiments strictly preserve the original 70:15:15 contiguous depth split, fit all preprocessing transformations exclusively on the 1,268 training rows, exclude the 397 unsupported transition rows from training loss, and retain Tests 20 and 21 exclusively for final physical evaluation.

### 9.1 Experiment A: Depth-Aware Models (+TVD_MSL)

#### Hypothesis and Architecture
Experiment A introduces True Vertical Depth Subsea (`TVD_MSL`, m MSL) from Trajectory 1 as a 14th predictor alongside the 13 baseline logs.

#### Results
- **MLP (Continuous Extrapolation):** By learning a continuous depth slope across the training interval, the MLP extrapolates into the deep Skagerrak, predicting Test 20 at **364.404 bar** (residual +4.302 bar) and Test 21 at **376.154 bar** (residual -7.336 bar), achieving a combined MAE of **5.819 bar** across Tests 20 & 21 (compared to 31.110 bar in the baseline).
- **Validation Penalty (Overfitting to Depth):** However, on the unseen lower Sleipner validation checkpoint (Test 16 at 3,324.65 m MD), the depth-aware MLP overpredicts at **345.354 bar** (residual -3.641 bar), degrading from its exceptional baseline error (+0.270 bar).
- **Tree Models (Plateauing at Training Depth Bound):** Decision tree models (Random Forest, CatBoost, Gradient Boosting) split on depth thresholds. For any test observation with $\text{TVD} > \max(\text{TVD}_{\text{train}})$, trees route observations to the terminal leaf of the deepest split. Consequently, GBR flatlines at **341.422 bar** across the entire validation and testing sequence, yielding a constant residual plateau (+27.384 bar on Tests 20 & 21).

#### Geological and Geomechanical Risk Analysis
- **Depth Memorization:** Depth-aware models substitute physical petrophysical relationships with a geometric coordinate. In a single borehole, depth strongly correlates with hydrostatic loading ($dP/dz \approx \rho g$). However, across structural unconformities, fault displacements, or lateral compartment boundaries, depth alone does not dictate pore pressure.
- **Poor Lateral Transferability:** If a depth-memorizing model is deployed in a neighboring offset well with a 50 m structural throw, it will erroneously predict overpressures at the wrong stratigraphic level. Depth awareness must not replace petrophysical logging responses.

---

### 9.2 Experiment B: Physics-Informed Features (+Sv, +Ph, +sigma_v_hydro, +DT_nct_ratio)

#### Feature Formulation and Provenance
Experiment B investigates four physically derived features (total 17 predictors) constructed without using any measured wellbore pressure stations:
1. **Total Vertical Overburden Stress ($S_v$, bar):**
   $$S_v(z) = P_{\text{water}} + P_{\text{shallow}} + \int_{z_{\text{shallow}}}^z \rho_b(z') g dz'$$
   *Inputs:* Trajectory 1 TVD MSL, seabed 91.0 m ($1.025\text{ g/cm}^3$), reference shallow sediment density $1.90\text{ g/cm}^3$, continuous wireline bulk density ($\rho_b$).
   *Uncertainty:* $\pm 1.26$ bar composite uncertainty across shallow sediment density bounds ($1.80$ to $1.95\text{ g/cm}^3$).
2. **Normal Brine Hydrostatic Reference Pressure ($P_h$, bar):**
   $$P_h(z) = \rho_{\text{seawater}} g z_{\text{seabed}} + \rho_{\text{brine}} g (z - z_{\text{seabed}})$$
   *Inputs:* Water depth 91.0 m, formation brine density $1.030\text{ g/cm}^3$, $g = 9.80665\text{ m/s}^2$.
   *Uncertainty:* $\pm 0.5$ bar across brine density range $1.025$ to $1.035\text{ g/cm}^3$.
3. **Nominal Hydrostatic Effective Stress ($\sigma_{v,\text{hydro}}$, bar):**
   $$\sigma_{v,\text{hydro}}(z) = S_v(z) - P_h(z)$$
   *Inputs:* Difference between lithostatic overburden and normal hydrostatic pressure.
4. **Regional Compaction Slowness Ratio ($DT / DT_{\text{nct}}$):**
   $$DT_{\text{nct}}(z) = DT_0 \exp(-c \cdot z),\quad DT_{\text{ratio}} = \frac{DT}{DT_{\text{nct}}}$$
   *Inputs:* Regional North Sea shale compaction trend parameters ($DT_0 = 195.0\ \mu\text{s/ft}, c = 0.00045\text{ m}^{-1}$). Values $> 1.0$ indicate acoustic undercompaction.

#### Results
- **MLR Substantial Improvement:** Introducing $S_v$ and $P_h$ enables MLR to scale with lithostatic loading, predicting Test 20 at **352.355 bar** and Test 21 at **358.686 bar** (Tests 20 & 21 MAE drops from **36.871 bar** in baseline to **13.242 bar**).
- **MLP Physics Guidance:** The physics-informed MLP accurately predicts Test 20 at **370.964 bar** (residual -2.258 bar) and Test 21 at **384.826 bar** (residual -16.008 bar, MAE = 9.133 bar), while preserving reasonable validation accuracy on Test 16 (predicted 345.964 bar).
- **Tree Models:** Trees capture non-linear interactions with effective stress; Random Forest validation error on Test 16 improves to **1.049 bar** (predicted 340.665 bar).

---

### 9.3 Experiment C: Formation-Aware Models (+Stratigraphy, +Hardgrounds)

#### Feature Formulation and Geological Provenance
Experiment C introduces four geological indicators (total 17 predictors):
1. `IS_HUGIN`: Binary indicator for Hugin Sandstone (MD $\le 3,263.5$ m). Established sequence stratigraphic boundary.
2. `IS_SLEIPNER`: Binary indicator for Sleipner Formation ($3,263.5 < \text{MD} \le 3,386.5$ m). Fluvial-deltaic sequence.
3. `IS_SKAGERRAK`: Binary indicator for Skagerrak Formation (MD $> 3,386.5$ m). **Geological interpretation** across the regional unconformity; no impermeable barrier is directly observed.
4. `CALCITE_HARDGROUND`: Binary indicator for cemented calcite hardground baffle intervals (MD 3,286.5-3,290.8 m, 3,296.8-3,299.2 m, 3,326.3-3,330.1 m).

#### Results and Critical Machine Learning Insight
- **The Unseen Categorical Level Trap:**
  In the training partition (MD $\le 3,320.8$ m), `IS_SKAGERRAK` is identically zero (0% representation). Consequently:
  - In linear models (MLR), `IS_SKAGERRAK` has zero variance in training, receiving a coefficient of zero.
  - In tree models (RF, CatBoost, GBR), no split can be made on a constant zero feature. When evaluated on the test set where `IS_SKAGERRAK == 1`, tree models route the data down the default branch, predicting 332.0 to 334.2 bar (worse than baseline!).
  - In neural networks (MLP), evaluating an untrained node activation produces extreme instability, causing MLP to severely underpredict at **301.884 bar** on Test 20 (residual +66.822 bar!).
- **Methodological Conclusion:** Categorical formation flags cannot extrapolate pore pressure into an uncalibrated formation without prior regional pressure calibration for that specific formation.

---

### 9.4 Comprehensive Cross-Experiment Benchmark Summary

#### Table 5: Physical Pretest Station Benchmarks Across All Experiments

| Experiment | Model | 15 Train Pretests RMSE (bar) | Val Test 16 Pred (bar) | Val Test 16 Residual (bar) | Test 20 Pred (bar) | Test 21 Pred (bar) | Tests 20 & 21 MAE (bar) | Tests 20 & 21 RMSE (bar) | Excluded Test 19 Diff (bar) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logs-Only)** | MLR | 0.989 | 335.822 | +5.892 | 333.921 | 329.860 | 36.871 | 36.930 | +27.854 |
| **Baseline (Logs-Only)** | Random Forest | 0.481 | 334.801 | +6.912 | 335.295 | 338.038 | 32.095 | 32.122 | +19.676 |
| **Baseline (Logs-Only)** | CatBoost | **0.283** | 335.233 | +6.480 | 336.867 | 338.341 | 31.158 | 31.165 | +19.373 |
| **Baseline (Logs-Only)** | Gradient Boosting | 0.307 | 335.458 | +6.255 | 336.586 | 340.123 | 30.408 | 30.456 | +17.591 |
| **Baseline (Logs-Only)** | MLP | 1.016 | **341.443** | **+0.270** | 335.323 | 339.982 | 31.110 | 31.193 | +17.732 |
| **Exp A (+TVD_MSL)** | MLR | 1.196 | 337.537 | +4.176 | 339.163 | 340.246 | 29.057 | 29.083 | +17.468 |
| **Exp A (+TVD_MSL)** | Random Forest | 0.311 | 338.346 | +3.368 | 338.512 | 340.358 | 29.327 | 29.351 | +17.356 |
| **Exp A (+TVD_MSL)** | CatBoost | 0.211 | 337.855 | +3.858 | 337.791 | 339.196 | 30.269 | 30.292 | +18.518 |
| **Exp A (+TVD_MSL)** | Gradient Boosting | **0.022** | 341.422 | +0.291 | 341.401 | 341.355 | 27.384 | 27.399 | +16.359 |
| **Exp A (+TVD_MSL)** | MLP | 0.964 | 345.354 | -3.641 | **364.404** | **376.154** | **5.819** | **6.035** | -18.440 |
| **Exp B (+Physics)** | MLR | 1.049 | 339.484 | +2.229 | 352.355 | 358.686 | 13.242 | 13.435 | -0.972 |
| **Exp B (+Physics)** | Random Forest | 0.218 | 340.665 | +1.049 | 340.536 | 340.961 | 28.014 | 28.026 | +16.753 |
| **Exp B (+Physics)** | CatBoost | 0.178 | 337.297 | +4.416 | 336.822 | 338.657 | 31.023 | 31.046 | +19.057 |
| **Exp B (+Physics)** | Gradient Boosting | **0.029** | 341.461 | +0.252 | 341.439 | 341.395 | 27.345 | 27.360 | +16.319 |
| **Exp B (+Physics)** | MLP | 1.049 | 345.964 | -4.251 | **370.964** | **384.826** | **9.133** | **11.424** | -27.112 |
| **Exp C (+Formation)** | MLR | 1.200 | 335.502 | +6.211 | 330.613 | 327.454 | 39.729 | 40.046 | +30.260 |
| **Exp C (+Formation)** | Random Forest | 0.093 | 335.051 | +6.662 | 332.036 | 334.184 | 35.652 | 35.733 | +23.530 |
| **Exp C (+Formation)** | CatBoost | 0.134 | 334.941 | +6.772 | 332.277 | 332.537 | 36.355 | 36.378 | +25.178 |
| **Exp C (+Formation)** | Gradient Boosting | 0.129 | 335.352 | +6.361 | 332.839 | 333.942 | 35.372 | 35.419 | +23.772 |
| **Exp C (+Formation)** | MLP | 0.472 | 338.514 | +3.199 | 301.884 | 319.101 | 58.270 | 60.180 | +38.613 |

*(Note: Residual = Actual minus Predicted in bar).*

#### Table 6: Partition Metrics Across Baseline and All Experiments (Valid Targets Only)

| Experiment | Model | Training RMSE (bar) | Training $R^2$ | Validation MAE (bar) | Validation RMSE (bar) | Test Anchors MAE (bar) | Test Prov. MAE (bar) | Full Valid RMSE (bar) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | MLR | 1.933 | 0.889 | 5.980 | 5.993 | 36.871 | 36.135 | 10.393 |
| **Baseline** | Random Forest | 0.332 | 0.997 | 6.588 | 6.614 | 32.095 | 33.174 | 9.398 |
| **Baseline** | CatBoost | **0.202** | **0.999** | 6.307 | 6.335 | 31.158 | 33.580 | 9.496 |
| **Baseline** | Gradient Boosting | 0.217 | 0.999 | 6.336 | 6.351 | 30.408 | 33.032 | 9.357 |
| **Baseline** | MLP | 1.495 | 0.934 | **2.707** | **3.588** | 31.110 | **31.247** | **8.974** |
| **Exp A (+TVD)**| MLR | 1.636 | 0.920 | 4.249 | 4.281 | 29.057 | 28.328 | 8.219 |
| **Exp A (+TVD)**| Random Forest | 0.165 | 0.999 | 3.428 | 3.490 | 29.327 | 30.509 | 8.577 |
| **Exp A (+TVD)**| CatBoost | 0.111 | 1.000 | 3.916 | 3.974 | 30.269 | 32.730 | 9.201 |
| **Exp A (+TVD)**| Gradient Boosting | **0.012** | **1.000** | **0.301** | **0.354** | 27.384 | 27.387 | 7.712 |
| **Exp A (+TVD)**| MLP | 1.455 | 0.937 | 3.708 | 3.750 | **5.819** | **7.545** | **2.880** |
| **Exp B (+Phys)**| MLR | 1.488 | 0.934 | 2.378 | 2.520 | 13.242 | 10.957 | 3.568 |
| **Exp B (+Phys)**| Random Forest | 0.117 | 1.000 | 1.139 | 1.341 | 28.014 | 29.239 | 8.225 |
| **Exp B (+Phys)**| CatBoost | 0.088 | 1.000 | 4.473 | 4.526 | 31.023 | 33.486 | 9.407 |
| **Exp B (+Phys)**| Gradient Boosting | **0.015** | **1.000** | **0.260** | **0.315** | 27.345 | 27.348 | 7.701 |
| **Exp B (+Phys)**| MLP | 1.454 | 0.937 | 4.316 | 4.353 | **9.133** | **15.158** | **4.791** |
| **Exp C (+Form)**| MLR | 1.916 | 0.891 | 6.299 | 6.311 | 39.729 | 41.517 | 11.839 |
| **Exp C (+Form)**| Random Forest | 0.063 | 1.000 | 6.662 | 6.685 | 35.652 | 36.721 | 10.370 |
| **Exp C (+Form)**| CatBoost | 0.082 | 1.000 | 6.772 | 6.794 | 36.355 | 38.802 | 10.923 |
| **Exp C (+Form)**| Gradient Boosting | **0.060** | **1.000** | 6.361 | 6.386 | 35.372 | 37.408 | 10.534 |
| **Exp C (+Form)**| MLP | 1.082 | 0.965 | **3.220** | **3.315** | 58.270 | 50.158 | 14.869 |

---

### 9.5 Synthesis and Modeling Recommendations

1. **Trade-Off of Depth Awareness (Exp A):**
   Adding depth drastically reduces Skagerrak extrapolation error in continuous neural models (Tests 20 & 21 MAE drops to 5.82 bar in MLP), but at the cost of overpredicting validation pressure by 3.64 bar and encouraging non-physical depth memorization.
2. **Value of Physics-Informed Features (Exp B):**
   Physics-informed features ($S_v, P_h, \sigma_{v,\text{hydro}}$) provide the soundest mechanism for scaling predictions into deeper regimes. MLR Test 20/21 MAE drops from 36.87 to 13.24 bar, and MLP reaches 9.13 bar MAE, without relying on pure depth memorization.
3. **The Categorical Trap of Formation Flags (Exp C):**
   Discrete formation indicators fail when an overpressured compartment is not present in training data. Unseen categorical levels cannot supply pressure offsets without explicit prior calibration.
4. **Final Architecture Guidance:**
   - **Reservoir in-distribution characterization:** CatBoost on baseline logs remains the most stable, highest-precision model (training RMSE = 0.202 bar).
   - **Transition step generalization:** MLP Regressor with physics features ($S_v, P_h$) offers the strongest combination of continuous step detection and bounded physical scaling.

---

## 10. Complete Deliverable Manifest

All baseline and experimental models, metrics, tables, and figures are dual-saved and verified:

- **Baseline Artifacts:**
  - `outputs/models/{mlr,random_forest,catboost,gradient_boosting,mlp}_model.joblib`
  - `outputs/models/log_standard_scaler.joblib`
  - `outputs/models/training_manifest.json`
  - `outputs/metrics/model_predictions_1811.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/partition_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/provenance_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/formation_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/station_evaluation_metrics.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/physical_station_benchmarks.csv` & `.json` (mirrored in `tables/`)
  - `outputs/metrics/feature_importance.csv` & `.json` (mirrored in `tables/`)
  - `outputs/figures/{actual_vs_predicted,residuals_vs_depth_and_dist,borehole_pressure_profiles,feature_importance_comparison,learning_curves}.png`
- **Experimental Artifacts (A, B, C):**
  - Models: `outputs/experiments/models/{exp_a,exp_b,exp_c}_{mlr,random_forest,catboost,gradient_boosting,mlp}_model.joblib`
  - Predictions: `outputs/experiments/metrics/{exp_a,exp_b,exp_c}_predictions.csv` & `.json` (mirrored in `tables/experiments/`)
  - Comparison Tables: `outputs/experiments/metrics/experiment_comparison_summary.csv` & `.json` (mirrored in `tables/experiments/`)
  - Station Benchmarks: `outputs/experiments/metrics/experiment_station_benchmarks.csv` & `.json` (mirrored in `tables/experiments/`)
  - Provenance Metrics: `outputs/experiments/metrics/experiment_provenance_metrics.csv` & `.json` (mirrored in `tables/experiments/`)
  - Feature Metadata: `outputs/experiments/metrics/experiment_{b,c}_*features_metadata.json`
  - Diagnostic Figure: `outputs/experiments/figures/all_experiments_borehole_comparison.png`
- **Automated Verification:**
  - `tests/test_model_training.py` (33 tests pass)
  - `tests/test_experiments.py` (4 tests pass; total test suite = 37 tests, 100% pass rate).


