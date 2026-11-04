# Repository and Architecture Audit Report

This report presents a formal technical audit of the Well 15/9-F-12 pore pressure prediction repository, verifying data integrity, geological constraints, geomechanical baselines, and modular pipeline architecture.

Date of Audit: 2026-11-04 18:03:37 UTC
Repository: wisdomdivine/final-year-project
Branch: main

---

## 1. Directory Scaffold and Project Architecture

The project has been structured into a production-grade, modular architecture:

- `configs/`: YAML configuration files defining dataset paths, feature definitions, depth partition boundaries, geomechanical targets, and model hyperparameters.
  - `configs/data.yaml`: Dataset specifications and continuous depth boundaries.
  - `configs/target_generation.yaml`: Overburden density parameters, hydrostatic fluids, and Eaton calibration.
  - `configs/models.yaml`: Hyperparameters for MLR, Random Forest, CatBoost, and MLPRegressor.
- `src/`: Core Python package containing reusable, tested scientific modules.
  - `src/data/`: Data loading, validation, feature scaling, and depth partitioning routines.
  - `src/physics/`: Minimum-curvature TVD trajectory calculation, overburden stress integration, hydrostatic reference calculation, NCT trend fitting, and Eaton pore pressure formulation.
  - `src/models/`: Standardized regressor wrappers for MLR, Random Forest, CatBoost, and MLP.
  - `src/evaluation/`: Metric computation (MAE, MSE, RMSE, R-squared, MAPE) and residual diagnostics.
  - `src/utils/`: Standardized dual-saving (CSV and JSON), structured logging, and seed reproducibility.
- `tests/`: Automated unit test suite run with pytest (`tests/test_data_validation.py`, `tests/test_physics_models.py`).
- `outputs/`: Dedicated artifact directories:
  - `outputs/audit/`: Audit reports, JSON validation payloads, predictor summaries, and reference tables.
  - `outputs/figures/`: High-resolution diagnostic figures.
  - `outputs/metrics/`: Model evaluation scores and performance comparisons.
  - `outputs/models/`: Serialized trained model objects.
  - `outputs/reports/`: Technical reports and documentation.
- `run.py`: Top-level command-line execution entry point.

---

## 2. Dataset Verification and Validation Status

The predictor dataset has been reconstructed from the primary composite well logs and validated against published benchmarks (Siddique et al.):

- Path: `dataset/F12_predictor_dataset_1811.csv`
- Total Observations: 1,811 rows
- Measured Depth Range: 3,127.7052 m to 3,403.5492 m MD
- Depth Step: 0.1524 m (strictly monotonic, strictly increasing)
- Predictor Variables (13 features + Depth):
  - Acoustic slowness: `DT` (mean 82.92 us/ft)
  - Radioactivity: `GR` (mean 64.19 API, range 22.45 to 122.87 API)
  - Porosity logs: `NPHI` (neutron), `RHOB` (bulk density, mean 2.39 g/cm3), `PHIF` (formation porosity)
  - Resistivity array: `RD` (deep), `RS` (shallow), `RT` (true formation resistivity)
  - Operational parameters: `ROP` (rate of penetration, mean 17.98 m/hr)
  - Compositional and fluid indicators: `BVW` (bulk volume water), `K` (permeability), `SW` (water saturation), `VCL` (clay volume)
- Data Quality Checks:
  - Total null / missing values: 0
  - Total infinite values: 0
  - Overall Validation Status: PASS

---

## 3. Contiguous Depth Partitioning (70:15:15)

To prevent spatial autocorrelation and data leakage typical of random k-fold cross-validation in wellbore data, the reservoir sequence is partitioned contiguously along depth:

1. Training Set (70.02%, 1,268 observations):
   - Depth range: 3,127.7052 m to 3,320.7960 m MD (2,853.25 m to 2,987.35 m TVD MSL)
   - Geological interval: Hugin Sandstone reservoir and upper Sleipner Formation.
   - Contains pressure-test stations 1 to 15.
2. Validation Set (15.02%, 272 observations):
   - Depth range: 3,320.9484 m to 3,362.2488 m MD (2,987.48 m to 3,010.36 m TVD MSL)
   - Geological interval: Middle to lower Sleipner Formation transition zone.
   - Contains pressure-test stations 16 and 17 (station 17 confirmed dry test).
3. Test Set (14.96%, 271 observations):
   - Depth range: 3,362.4012 m to 3,403.5492 m MD (3,010.45 m to 3,017.83 m TVD MSL)
   - Geological interval: Deep Skagerrak Formation overpressured regime.
   - Contains pressure-test stations 18 to 21 (companion repeat pairs 18/20 and 19/21).

---

## 4. Geomechanical and Hydrostatic References

A defensible physical target requires precise overburden and hydrostatic baselines:

1. Total Overburden Stress Sv(z):
   - Reconstructed by direct vertical integration of continuous bulk density from 262.43 m TVD MSL down to 3,017.83 m TVD MSL.
   - Truly unmeasured shallow sediment is only 171.43 m (seabed 91.0 m to 262.43 m TVD MSL), constraining shallow density uncertainty to +/- 1.26 bar (+/- 0.21%).
   - Sv at top of predictor interval (3,127.71 m MD): 580.05 bar (8,413 psi, OBG = 0.2033 bar/m, 2.073 s.g.).
   - Sv at base of predictor interval (3,403.55 m MD): 618.07 bar (8,964 psi, OBG = 0.2048 bar/m, 2.088 s.g.).
2. Normal Hydrostatic Reference Ph(z):
   - Seawater column: 9.15 bar at seabed (91.0 m TVD MSL).
   - Normal formation brine hydrostatic gradient: 0.1010 bar/m (fluid density 1.030 g/cm3).
   - Hydrostatic pressure at top predictor (2,853.25 m TVD MSL): 288.42 bar.
   - Hydrostatic pressure at base predictor (3,017.83 m TVD MSL): 305.06 bar.
3. Pressure-Regime Audit:
   - Hugin interval (3,127.7 m to 3,263.5 m MD): Hydrostatic oil column with fitted gradient 0.07028 bar/m (R2 = 0.999915, oil density 0.7167 g/cm3, 42.6 deg API).
   - Sleipner interval (3,293.6 m to 3,324.7 m MD): Baffled compartmentalized steps (334.10 bar to 341.71 bar).
   - Skagerrak interval (3,386.5 m to 3,403.6 m MD): Fully stabilized overpressure regime at 368.71 bar to 368.82 bar.

---

## 5. Automated Test Suite

Automated verification is implemented using pytest across two core test modules:

1. `tests/test_data_validation.py`:
   - Shape and column completeness.
   - Zero nulls and zero infinite values.
   - Strict depth monotonicity and 0.1524 m sampling step.
   - Comprehensive validation runner status.
   - Non-overlapping depth partitioning counts.
   - Data leakage prevention during scaling.
2. `tests/test_physics_models.py`:
   - Minimum curvature trajectory computation (vertical and horizontal extremes).
   - Analytical hydrostatic reference pressure at seabed and deep reservoir.
   - Overburden vertical integration against analytical density slab.
   - Physical bounds of Eaton formulation (hydrostatic equality, undercompaction, lithostatic limit).
   - Normal Compaction Trend fitting and prediction.

All 12 automated test cases pass with zero failures.
