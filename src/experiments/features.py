"""Feature construction routines for controlled modeling experiments.

Experiments:
1. Experiment A (Depth-Aware):
   Adds TVD_MSL (Trajectory 1, m MSL) to the baseline 13 petrophysical features.
2. Experiment B (Physics-Informed):
   Adds total vertical stress Sv(z), normal brine hydrostatic pressure Ph(z),
   nominal hydrostatic effective stress sigma_v_hydro(z), and acoustic slowness ratio
   DT / DT_nct based on regional shale normal compaction trend.
3. Experiment C (Formation-Aware):
   Adds independent lithostratigraphic indicators (IS_HUGIN, IS_SLEIPNER, IS_SKAGERRAK)
   and calcite hardground cementing baffle flags (CALCITE_HARDGROUND).

Zero held-out pressure measurements are used in any feature derivation.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data.validation import EXPECTED_PREDICTORS
from src.physics.hydrostatic import calculate_hydrostatic_pressure
from src.physics.nct import predict_normal_transit_time

# Regional Normal Compaction Trend parameters (from regional North Sea studies / config)
REGIONAL_DT0_US_FT = 195.0
REGIONAL_COMPACTION_C = 0.00045

# Hardground cementing intervals in Sleipner Formation (MD, m)
CALCITE_HARDGROUND_INTERVALS = [
    (3286.5, 3290.8),
    (3296.8, 3299.2),
    (3326.3, 3330.1),
]

# Formation boundary depths (MD, m) from well completion report
HUGIN_BASE_MD = 3263.5
SLEIPNER_BASE_MD = 3386.5


def build_experiment_a_features(
    df_pred: pd.DataFrame,
    df_targ: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct Experiment A feature matrix:
    Baseline 13 logs + TVD_MSL (Trajectory 1). Total: 14 predictors.
    """
    features = list(EXPECTED_PREDICTORS)
    df_out = df_pred[features].copy()
    df_out["TVD_MSL"] = df_targ["TVD_MSL"].values.astype(float)
    return df_out


def build_experiment_b_features(
    df_pred: pd.DataFrame,
    df_targ: pd.DataFrame,
    ovb_sensitivity_df: pd.DataFrame,
    include_dt_ratio: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Construct Experiment B feature matrix:
    Baseline 13 logs + 3 or 4 physically derived features.
    Total: 17 predictors if include_dt_ratio is True, or 16 predictors if False.

    Derived features:
    1. SV_OVERBURDEN_BAR: Total vertical overburden stress Sv (bar) integrated
       along Trajectory 1 with reference shallow sediment density 1.90 g/cm3.
    2. PH_HYDROSTATIC_BAR: Normal brine hydrostatic reference pressure Ph (bar)
       assuming seabed at 91 m TVD MSL (seawater 1.025 g/cm3) and formation brine 1.030 g/cm3.
    3. SIGMA_V_HYDRO_BAR: Nominal hydrostatic effective stress Sv - Ph (bar).
    4. DT_NCT_RATIO (optional): Slowness compaction ratio DT / DT_nct(TVD_MSL) relative to
       regional North Sea normal compaction baseline DT_nct = 195 * exp(-0.00045 * TVD).

    Documentation metadata dictionary returned alongside the DataFrame.
    """
    features = list(EXPECTED_PREDICTORS)
    df_out = df_pred[features].copy()

    tvd_msl = df_targ["TVD_MSL"].values.astype(float)

    # 1. Overburden total vertical stress
    sv_bar = ovb_sensitivity_df["Sv_traj1_rho1.9_bar"].values.astype(float)
    df_out["SV_OVERBURDEN_BAR"] = sv_bar

    # 2. Normal hydrostatic reference pressure
    ph_bar = calculate_hydrostatic_pressure(
        tvd_msl,
        water_depth_m=91.0,
        seawater_density_gcc=1.025,
        formation_fluid_density_gcc=1.030,
    )
    df_out["PH_HYDROSTATIC_BAR"] = ph_bar

    # 3. Nominal hydrostatic effective stress
    sigma_v = sv_bar - ph_bar
    df_out["SIGMA_V_HYDRO_BAR"] = sigma_v

    metadata = {
        "SV_OVERBURDEN_BAR": {
            "equation": "Sv(z) = P_water + P_shallow + integral(rhob * g * dz)",
            "inputs": "Trajectory 1 TVD MSL, seawater depth 91.0 m (1.025 g/cm3), shallow density 1.90 g/cm3, bulk density log (RHOB)",
            "assumptions": "Vertical principle stress alignment; reference shallow density 1.90 g/cm3 for 91-262 m TVD MSL",
            "uncertainty": "+/- 1.26 bar (+/- 0.21%) across shallow density range 1.80 to 1.95 g/cm3",
            "pressure_data_used": "None (pure density integration)",
        },
        "PH_HYDROSTATIC_BAR": {
            "equation": "Ph(z) = P_seabed + rho_brine * g * (z - z_water)",
            "inputs": "Trajectory 1 TVD MSL, seabed 91.0 m, seawater 1.025 g/cm3, brine 1.030 g/cm3",
            "assumptions": "Continuous normally pressured formation water column; uniform salinity",
            "uncertainty": "+/- 0.5 bar across formation brine density range 1.025 to 1.035 g/cm3",
            "pressure_data_used": "None (pure hydrostatic column)",
        },
        "SIGMA_V_HYDRO_BAR": {
            "equation": "sigma_v_hydro(z) = Sv(z) - Ph(z)",
            "inputs": "SV_OVERBURDEN_BAR, PH_HYDROSTATIC_BAR",
            "assumptions": "Nominal vertical effective stress under hypothetical normal pore fluid pressure",
            "uncertainty": "+/- 1.4 bar composite uncertainty from Sv and Ph",
            "pressure_data_used": "None",
        },
    }

    # 4. Regional Normal Compaction Trend transit time and compaction ratio (optional)
    if include_dt_ratio:
        dt_nct = predict_normal_transit_time(
            tvd_msl,
            dt0_us_ft=REGIONAL_DT0_US_FT,
            compaction_coeff_c=REGIONAL_COMPACTION_C,
        )
        dt_ratio = df_pred["DT"].values / dt_nct
        df_out["DT_NCT_RATIO"] = dt_ratio

        metadata["DT_NCT_RATIO"] = {
            "equation": "DT_ratio = DT / (DT0 * exp(-c * TVD_MSL))",
            "inputs": "Sonic slowness DT (us/ft), Trajectory 1 TVD MSL, regional DT0=195.0 us/ft, c=0.00045 m^-1",
            "assumptions": "Regional mechanical compaction trend in pure mudstones; ratio > 1.0 indicates undercompaction",
            "uncertainty": "+/- 5 us/ft in baseline DT0; chemical cementation in sandstones creates structural divergence",
            "pressure_data_used": "None (regional geological constants only)",
        }

    return df_out, metadata

    return df_out, metadata


def build_experiment_c_features(
    df_pred: pd.DataFrame,
    df_targ: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Construct Experiment C feature matrix:
    Baseline 13 logs + 3 lithostratigraphic indicators + 1 calcite hardground baffle indicator.
    Total: 17 predictors.

    Features:
    1. IS_HUGIN: 1 if MD <= 3263.5 m, else 0 (Hugin Sandstone reservoir).
    2. IS_SLEIPNER: 1 if 3263.5 < MD <= 3386.5 m, else 0 (Sleipner Formation).
    3. IS_SKAGERRAK: 1 if MD > 3386.5 m, else 0 (Skagerrak Formation).
    4. CALCITE_HARDGROUND: 1 if MD falls inside recognized calcite-cemented hardground zones.

    Note: The Sleipner/Skagerrak boundary at 3,386.5 m MD is an interpretation from
    regional biostratigraphy and cuttings lithology across a transitional facies.
    In the training partition (MD <= 3,320.8 m), IS_SKAGERRAK is all zeros.
    """
    features = list(EXPECTED_PREDICTORS)
    df_out = df_pred[features].copy()

    depths = df_targ["DEPTH"].values.astype(float)

    is_hugin = (depths <= HUGIN_BASE_MD).astype(float)
    is_sleipner = ((depths > HUGIN_BASE_MD) & (depths <= SLEIPNER_BASE_MD)).astype(float)
    is_skagerrak = (depths > SLEIPNER_BASE_MD).astype(float)

    is_hardground = np.zeros_like(depths, dtype=float)
    for top_hg, base_hg in CALCITE_HARDGROUND_INTERVALS:
        mask_hg = (depths >= top_hg) & (depths <= base_hg)
        is_hardground[mask_hg] = 1.0

    df_out["IS_HUGIN"] = is_hugin
    df_out["IS_SLEIPNER"] = is_sleipner
    df_out["IS_SKAGERRAK"] = is_skagerrak
    df_out["CALCITE_HARDGROUND"] = is_hardground

    metadata = {
        "IS_HUGIN": {
            "definition": "1 if MD <= 3263.5 m, else 0",
            "source": "Well 15/9-F-12 completion report / NPD lithostratigraphy",
            "certainty": "Established sequence stratigraphic boundary (sharp gamma ray and core contact)",
            "training_coverage": "Rows 0 to 891 (100% within training set)",
        },
        "IS_SLEIPNER": {
            "definition": "1 if 3263.5 < MD <= 3386.5 m, else 0",
            "source": "Well 15/9-F-12 completion report / NPD lithostratigraphy",
            "certainty": "Flurial-deltaic sequence containing coal, siltstone, and discrete sandstone bodies",
            "training_coverage": "Rows 892 to 1267 (partially in training, partially in validation and transition)",
        },
        "IS_SKAGERRAK": {
            "definition": "1 if MD > 3386.5 m, else 0",
            "source": "Regional biostratigraphic correlation across regional unconformity",
            "certainty": "Geological interpretation across interbedded transition; no physical impermeable seal directly observed",
            "training_coverage": "0% in training set (all zeros; unseen category during model fitting)",
        },
        "CALCITE_HARDGROUND": {
            "definition": "1 if within 3286.5-3290.8 m, 3296.8-3299.2 m, or 3326.3-3330.1 m MD, else 0",
            "source": "Core description and high-density / acoustic spikes on wireline logs",
            "certainty": "Observed petrophysical / diagenetic cementation zones creating hydraulic baffles",
            "training_coverage": "Intervals 1 and 2 covered in training set; interval 3 in validation transition",
        },
    }

    return df_out, metadata

