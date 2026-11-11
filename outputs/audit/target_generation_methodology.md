# Target-Generation Methodology and Geomechanical Calibration

This document establishes the technical, mathematical, and petrophysical specification for constructing formation pore-pressure target vectors in Well 15/9-F-12.

Date: 2026-11-11
Repository: wisdomdivine/final-year-project
Status: Technical Specification for Review Prior to Vector Generation

---

## 1. Physical Regime Classification and Interval Formulations

The 1,811-observation study interval (3,127.7052 m to 3,403.5492 m MD) traverses three distinct geological formations with contrasting fluid types, compaction states, and pressure regimes.

### 1.1 Hugin Formation (3,127.7052 m to 3,263.5000 m MD / Top Reservoir to 3,263.5 m MD)
- Lithology: Clean, porous, highly permeable reservoir sandstone (SAND_FLAG = 1.0, mean porosity 0.22 v/v, mean permeability 512 mD).
- Direct Physical Measurements: 12 Schlumberger Run 10 MWD pretest stations (Tests 1 to 12).
- Fluid Regime: Continuous, connected hydrostatic hydrocarbon (oil) column.
- Mathematical Formulation:
  P_form(z) = 118.205 + 0.07028 * TVD_MSL (bar)
- Fluid Gradient Validation:
  - Fitted slope: 0.07028 bar/m
  - Equivalent fluid density: rho_oil = 0.07028 / 0.0980665 = 0.7167 g/cm3
  - Equivalent API gravity: 42.6 deg API
  - Coefficient of determination: R2 = 0.999915
  - Max residual across all 12 test stations: 0.038 bar
- Value Classification:
  - At pressure-test stations: Directly measured physical formation pressure.
  - Between stations in sandstone: Rigorously calculated from measured oil gradient.
  - Intercalated thin mudstones: Mechanically compacted, near-equilibrium with reservoir sand.

### 1.2 Sleipner Formation (3,263.5000 m to 3,340.0000 m MD)
- Lithology: Interbedded fluvial sandstones, carbonaceous shales, and thin calcite-cemented stringers.
- Direct Physical Measurements:
  - Test 13/14 (3,293.8 m MD / 2,915.4 m TVD MSL): 334.101 bar (stabilized companion repeat).
  - Test 15 (3,307.7 m MD / 2,923.7 m TVD MSL): 335.610 bar.
  - Test 16 (3,324.7 m MD / 2,933.8 m TVD MSL): 341.713 bar.
  - Test 17 (3,333.6 m MD / 2,939.4 m TVD MSL): 69.279 bar (dry test in tight mudstone; excluded from calibration).
- Fluid Regime: Baffled, compartmentalized progression.
  - Sand intervals are segmented by continuous carbonate hardgrounds (CARB_FLAG = 1.0, density 2.50 to 2.66 g/cm3, DT 64 to 75 us/ft) identified at:
    - 3,286.5 to 3,290.8 m MD
    - 3,296.8 to 3,299.2 m MD
    - 3,326.3 to 3,330.1 m MD
  - These low-permeability hardgrounds act as vertical pressure baffles, maintaining discrete step increases in pressure across the Sleipner sand bodies.
- Mathematical Formulation:
  - Piecewise compartment anchoring: Each sandstone lens is anchored to its respective verified pretest measurement (334.10 bar, 335.61 bar, 341.71 bar).
  - Internal fluid gradients within individual sand bodies: 0.0981 to 0.1010 bar/m (formation brine / light hydrocarbon mixture).
- Value Classification:
  - At stations: Measured.
  - Within sand bodies: Calculated from local fluid density.
  - Intervening shales: Estimated using calibrated effective stress balance.

### 1.3 Skagerrak Formation (3,340.0000 m to 3,403.5492 m MD)
- Lithology: Deep, overpressured Triassic sandstones and silty shales.
- Direct Physical Measurements:
  - Test 18 (3,386.51 m MD): 354.001 bar (ASC Quality 0: initial incomplete pretest).
  - Test 20 (3,386.99 m MD): 368.706 bar (ASC Quality 2 / report Good Test: stabilized companion repeat).
  - Test 19 (3,403.53 m MD): 357.714 bar (ASC Quality 0: tool shutdown, recovered from memory).
  - Test 21 (3,403.57 m MD): 368.818 bar (ASC Quality 2 / report Good Test: stabilized companion repeat).
- Fluid Regime: Deep overpressured reservoir compartment.
  - Stabilized tests 20 and 21 demonstrate that the active reservoir pore pressure is 368.71 to 368.82 bar.
  - The pressure differential between Test 20 (368.71 bar at 3,026.3 m TVD RKB) and Test 21 (368.82 bar at 3,036.3 m TVD RKB) is 0.112 bar over 10.0 m TVD, representing an internally consistent fluid gradient of 0.0112 bar/m.
  - Overpressure Magnitude: Overpressure relative to normal hydrostatic brine (305.06 bar) is +63.76 bar (+925 psi).
- Value Classification:
  - At stations 20 and 21: Measured.
  - Across the Skagerrak test partition: Interpolated/calculated compartment pressure.

---

## 2. Eaton Normal Compaction Trend (NCT) Analysis and Limitations

The Eaton (1975) sonic equation relates formation pore pressure to acoustic transit time:

Pp(z) = Sv(z) - [Sv(z) - Ph(z)] * [DT_n(z) / DT_obs(z)]^x

Where:
- Sv(z): Total vertical overburden stress (bar).
- Ph(z): Normal hydrostatic pressure (bar).
- DT_n(z): Normal compaction trend acoustic transit time in pure shale (us/ft).
- DT_obs(z): Observed compressional acoustic transit time (us/ft).
- x: Eaton sonic exponent (nominally 3.0 in disequilibrium-compaction Gulf of Mexico basins).

### 2.1 Critical Lithological and Physical Limitations in Well 15/9-F-12
1. Lithology Mismatch:
   - Eaton's equation was derived exclusively for thick, undercompacted marine shales where pore pressure elevation is generated by disequilibrium compaction (trapped pore fluids supporting overburden load).
   - In Well 15/9-F-12, 22 of the 23 valid pressure-test stations are in reservoir sandstone (SAND_FLAG = 1.0). Only Test 17 was attempted in shale/mudstone, and it resulted in a dry test (tight formation, 69.28 bar).
   - Reservoir sandstone pore pressure is controlled by fluid column buoyancy, structural entrapment, and regional compartment seals, not by local mechanical undercompaction. Fitting an Eaton trend directly to sandstone DT readings violates the fundamental physical premise of the method.

2. Chemical Compaction versus Mechanical Compaction:
   - At reservoir depths in the Volve Field (TVD > 2,800 m, temperatures > 70 deg C), Central North Sea shales have transitioned from mechanical compaction to chemical compaction (quartz cementation and illitization of smectite).
   - Chemical compaction permanently stiffens the shale rock frame, driving compressional transit time DT down to 75 to 82 us/ft.
   - When a shallow mechanical NCT (extrapolating DT_n from surface values of 190 to 200 us/ft down to ~114 us/ft at 2,850 m TVD) is applied to chemically cemented shales:
     DT_n(z) / DT_obs(z) = 114 / 78 = 1.46 > 1.0
   - Because DT_n / DT_obs > 1.0, raising this ratio to exponent x = 3.0 yields [DT_n / DT_obs]^3 = 3.12, causing the effective stress term to exceed the overburden stress. This produces unphysical, negative, sub-hydrostatic pore pressures:
     Pp = Sv - (Sv - Ph) * 3.12 << Ph (unphysical).

3. Methodological Resolution:
   - Eaton calibration must only be applied to shale intervals (VCL > 0.50, SAND_FLAG = 0, CARB_FLAG = 0).
   - Reservoir sandstones must be anchored directly to physical pretest measurements and validated hydrostatic fluid columns.
   - A hybrid, formation-aware target formulation must be utilized rather than a blanket, uncalibrated Eaton sonic profile.

---

## 3. Boundary Transitions and Compartment Handling

Formation boundaries represent geological discontinuities rather than smooth mathematical curves.

1. Hugin-Sleipner Boundary (3,263.50 m MD):
   - Hydrocarbon column terminates at the base of the Hugin sand.
   - Pressure increases across the boundary from 325.70 bar (base Hugin) to 334.10 bar (upper Sleipner sand), representing an 8.40 bar compartment step across the regional seal.
   - Boundary transition is modeled as a localized step-change across the boundary mudstone interval rather than a linear interpolation that would imply a non-physical fluid gradient.

2. Sleipner-Skagerrak Boundary (3,340.00 m MD):
   - Overpressure increases sharply from 341.71 bar (lower Sleipner) to 368.71 bar (Skagerrak), an increase of 27.00 bar across approximately 16 m of section.
   - The Skagerrak overpressure is maintained by dense calcite cementation and regional sealing shales at the base of the Sleipner Formation.

---

## 4. Prevention of Target Leakage and Circular Validation

To maintain rigorous machine learning validation standards:

1. Partition Isolation:
   - The dataset is partitioned contiguously along depth (Train: 3,127.71 to 3,320.80 m MD; Val: 3,320.95 to 3,362.25 m MD; Test: 3,362.40 to 3,403.55 m MD).
   - Training targets are generated using data exclusively within the training interval (Tests 1 to 15).
   - Test partition measurements (Tests 18, 19, 20, 21) must NEVER be used to calibrate model parameters, feature scalers, or Eaton exponents applied to the training set.
   - Target generation parameters for the test partition serve solely as ground-truth evaluation references, preventing circular leakage.

2. Feature Scaler Fitting:
   - All standard scalers, min-max scalers, and log transformations must be fitted strictly on X_train and applied forward to X_val and X_test without refitting.

3. Independent Validation Summary:
   - Target values in Hugin: Independently validated by oil density physics (42.6 deg API).
   - Target values in Sleipner: Measured at discrete stations; shale intervals require effective stress modeling.
   - Target values in Skagerrak: Independently validated by companion repeat agreement (Tests 20 and 21 within 0.11 bar).
