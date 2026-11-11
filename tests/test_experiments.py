"""Unit tests for controlled experiments A, B, and C."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.validation import EXPECTED_PREDICTORS
from src.experiments.features import (
    build_experiment_a_features,
    build_experiment_b_features,
    build_experiment_c_features,
)

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shared_data():
    df_pred = pd.read_csv(BASE_DIR / "dataset" / "F12_predictor_dataset_1811.csv")
    df_targ = pd.read_csv(BASE_DIR / "dataset" / "F12_targets_1811.csv")
    ovb_sens_df = pd.read_csv(BASE_DIR / "outputs" / "audit" / "overburden_sensitivity.csv")
    return df_pred, df_targ, ovb_sens_df


def test_experiment_a_feature_integrity(shared_data):
    """Verify Experiment A has 14 features (13 baseline logs + TVD_MSL)."""
    df_pred, df_targ, _ = shared_data
    df_exp_a = build_experiment_a_features(df_pred, df_targ)

    assert len(df_exp_a) == 1811
    assert len(df_exp_a.columns) == 14
    assert "TVD_MSL" in df_exp_a.columns
    for pred in EXPECTED_PREDICTORS:
        assert pred in df_exp_a.columns

    # No forbidden target leakage columns
    assert "PP_TARGET_BAR" not in df_exp_a.columns
    assert "MEASURED_PRESSURE_BAR" not in df_exp_a.columns
    assert "TARGET_PROVENANCE" not in df_exp_a.columns


def test_experiment_b_physics_feature_integrity(shared_data):
    """Verify Experiment B has 17 features and zero target pressure leakage."""
    df_pred, df_targ, ovb_sens_df = shared_data
    df_exp_b, meta = build_experiment_b_features(df_pred, df_targ, ovb_sens_df, include_dt_ratio=True)

    assert len(df_exp_b) == 1811
    assert len(df_exp_b.columns) == 17

    expected_physics_cols = [
        "SV_OVERBURDEN_BAR",
        "PH_HYDROSTATIC_BAR",
        "SIGMA_V_HYDRO_BAR",
        "DT_NCT_RATIO",
    ]
    for col in expected_physics_cols:
        assert col in df_exp_b.columns
        assert col in meta
        assert meta[col]["pressure_data_used"] in ("None", "None (pure density integration)", "None (pure hydrostatic column)", "None (regional geological constants only)")

    # Check physical monotonicity
    assert (df_exp_b["SV_OVERBURDEN_BAR"].diff().dropna() > 0).all()
    assert (df_exp_b["PH_HYDROSTATIC_BAR"].diff().dropna() > 0).all()
    assert (df_exp_b["SIGMA_V_HYDRO_BAR"] > 0).all()
    assert (df_exp_b["DT_NCT_RATIO"] > 0).all()


def test_experiment_b_no_dtratio_feature_integrity(shared_data):
    """Verify Experiment B without DT_ratio has 16 features."""
    df_pred, df_targ, ovb_sens_df = shared_data
    df_exp_b_no_dt, meta = build_experiment_b_features(df_pred, df_targ, ovb_sens_df, include_dt_ratio=False)

    assert len(df_exp_b_no_dt) == 1811
    assert len(df_exp_b_no_dt.columns) == 16
    assert "DT_NCT_RATIO" not in df_exp_b_no_dt.columns
    assert "DT_NCT_RATIO" not in meta

    for col in ["SV_OVERBURDEN_BAR", "PH_HYDROSTATIC_BAR", "SIGMA_V_HYDRO_BAR"]:
        assert col in df_exp_b_no_dt.columns
        assert col in meta


def test_experiment_c_formation_feature_integrity(shared_data):
    """Verify Experiment C has 17 features and IS_SKAGERRAK is 0 in training set."""
    df_pred, df_targ, _ = shared_data
    df_exp_c, meta = build_experiment_c_features(df_pred, df_targ)

    assert len(df_exp_c) == 1811
    assert len(df_exp_c.columns) == 17

    expected_formation_cols = [
        "IS_HUGIN",
        "IS_SLEIPNER",
        "IS_SKAGERRAK",
        "CALCITE_HARDGROUND",
    ]
    for col in expected_formation_cols:
        assert col in df_exp_c.columns
        assert col in meta

    # Check training partition isolation: IS_SKAGERRAK must be 0 in training
    train_mask = df_targ["PARTITION"] == "Training"
    assert (df_exp_c.loc[train_mask, "IS_SKAGERRAK"] == 0.0).all()
    assert (df_exp_c.loc[train_mask, "IS_HUGIN"].sum() + df_exp_c.loc[train_mask, "IS_SLEIPNER"].sum() == len(df_exp_c.loc[train_mask]))


def test_experiment_artifacts_and_metrics_saved():
    """Verify experimental models, metrics tables, and diagnostic figures exist."""
    exp_models_dir = BASE_DIR / "outputs" / "experiments" / "models"
    exp_metrics_dir = BASE_DIR / "outputs" / "experiments" / "metrics"
    exp_figures_dir = BASE_DIR / "outputs" / "experiments" / "figures"
    exp_tables_dir = BASE_DIR / "tables" / "experiments"

    for exp in ["exp_a", "exp_b", "exp_b_no_dtratio", "exp_c"]:
        for model in ["mlr", "random_forest", "catboost", "gradient_boosting", "mlp"]:
            artifact = exp_models_dir / f"{exp}_{model}_model.joblib"
            assert artifact.exists(), f"Missing artifact: {artifact}"

        pred_csv = exp_metrics_dir / f"{exp}_predictions.csv"
        assert pred_csv.exists()
        assert (exp_tables_dir / f"{exp}_predictions.csv").exists()

    summary_csv = exp_metrics_dir / "experiment_comparison_summary.csv"
    assert summary_csv.exists()
    assert (exp_tables_dir / "experiment_comparison_summary.csv").exists()

    st_bench_csv = exp_metrics_dir / "experiment_station_benchmarks.csv"
    assert st_bench_csv.exists()
    assert (exp_tables_dir / "experiment_station_benchmarks.csv").exists()

    fig_file = exp_figures_dir / "all_experiments_borehole_comparison.png"
    assert fig_file.exists()
    assert fig_file.stat().st_size > 10_000


def test_individual_physical_station_errors_artifacts():
    """Verify individual physical station errors and benchmark matrix tables exist and are valid."""
    exp_metrics_dir = BASE_DIR / "outputs" / "experiments" / "metrics"
    exp_tables_dir = BASE_DIR / "tables" / "experiments"

    csv_path = exp_metrics_dir / "individual_physical_station_errors.csv"
    assert csv_path.exists()
    assert (exp_tables_dir / "individual_physical_station_errors.csv").exists()

    df_errors = pd.read_csv(csv_path)
    assert len(df_errors) == 100  # 5 configs * 5 models * 4 stations

    # Verify all 4 required stations are present
    assert set(df_errors["test_no"].unique()) == {16, 19, 20, 21}

    # Verify Test 19 role is explicitly distinguished as memory observation
    t19_rows = df_errors[df_errors["test_no"] == 19]
    assert (t19_rows["station_role"] == "Excluded Conflicting Memory Pretest").all()
    assert (t19_rows["actual_pressure_bar"] == 357.714).all()

    # Verify wide matrix
    mat_csv = exp_metrics_dir / "individual_station_benchmark_matrix.csv"
    assert mat_csv.exists()
    assert (exp_tables_dir / "individual_station_benchmark_matrix.csv").exists()
    df_mat = pd.read_csv(mat_csv)
    assert len(df_mat) == 25  # 5 configs * 5 models


def test_mlp_depth_sensitivity_profile_artifacts():
    """Verify MLP depth sensitivity profile exists and demonstrates super-linear acceleration."""
    exp_metrics_dir = BASE_DIR / "outputs" / "experiments" / "metrics"
    exp_tables_dir = BASE_DIR / "tables" / "experiments"

    csv_path = exp_metrics_dir / "mlp_depth_sensitivity_profile.csv"
    assert csv_path.exists()
    assert (exp_tables_dir / "mlp_depth_sensitivity_profile.csv").exists()

    df_sens = pd.read_csv(csv_path)
    assert len(df_sens) == 251

    # Verify deep gradient exceeds lithostatic gradient (~0.22 bar/m)
    deep_rows = df_sens[df_sens["TVD_MSL_m"] >= 2975.0]
    assert (deep_rows["dP_dTVD_bar_per_m"] > 0.50).all()

