"""Unit tests for the model training, evaluation, and diagnostics pipeline."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.pipeline import (
    prepare_training_data,
    FORBIDDEN_PREDICTOR_COLS,
)
from src.data.validation import EXPECTED_PREDICTORS


BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data_bundle():
    """Load and prepare training data bundle."""
    return prepare_training_data(BASE_DIR)


def test_anti_leakage_predictor_matrix(data_bundle):
    """Verify that predictor matrices contain strictly the 13 petrophysical logs and zero leakage columns."""
    predictor_cols = data_bundle["predictor_cols"]
    assert len(predictor_cols) == 13
    assert predictor_cols == EXPECTED_PREDICTORS

    for col in FORBIDDEN_PREDICTOR_COLS:
        assert col not in predictor_cols
        assert col not in data_bundle["X_train_raw"].columns
        assert col not in data_bundle["X_train_log_scaled"].columns
        assert col not in data_bundle["X_val_raw"].columns
        assert col not in data_bundle["X_test_raw"].columns


def test_contiguous_split_and_valid_target_counts(data_bundle):
    """Verify 70:15:15 contiguous split counts and valid target masks."""
    assert len(data_bundle["X_train_raw"]) == 1268
    assert len(data_bundle["X_val_raw"]) == 272
    assert len(data_bundle["X_test_raw"]) == 271

    assert int(data_bundle["train_valid_mask"].sum()) == 1268
    assert int(data_bundle["val_valid_mask"].sum()) == 34
    assert int(data_bundle["test_valid_mask"].sum()) == 112
    assert int(data_bundle["valid_mask"].sum()) == 1414


def test_scaler_fitted_exclusively_on_training(data_bundle):
    """Verify log-StandardScaler is fitted strictly on the 1,268 training rows."""
    X_tr_sc = data_bundle["X_train_log_scaled"]
    X_val_sc = data_bundle["X_val_log_scaled"]
    X_te_sc = data_bundle["X_test_log_scaled"]

    # Training columns must have zero mean and unit population std
    tr_means = X_tr_sc.mean(axis=0).values
    tr_stds = X_tr_sc.std(axis=0, ddof=0).values
    assert np.allclose(tr_means, 0.0, atol=1e-6)
    assert np.allclose(tr_stds, 1.0, atol=1e-6)

    # Validation and Test sets must NOT be centered at zero mean
    assert not np.allclose(X_val_sc.mean(axis=0).values, 0.0, atol=1e-2)
    assert not np.allclose(X_te_sc.mean(axis=0).values, 0.0, atol=1e-2)


def test_trained_model_artifacts_and_manifest():
    """Verify all 5 model artifacts, scaler artifact, and manifest exist."""
    models_dir = BASE_DIR / "outputs" / "models"
    expected_keys = ["mlr", "random_forest", "catboost", "gradient_boosting", "mlp"]

    for key in expected_keys:
        artifact = models_dir / f"{key}_model.joblib"
        assert artifact.exists(), f"Missing model artifact: {artifact}"
        assert artifact.stat().st_size > 100

    scaler_path = models_dir / "log_standard_scaler.joblib"
    assert scaler_path.exists()

    manifest_path = models_dir / "training_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["random_seed"] == 42
    assert manifest["data_integrity_verification"]["scaler_fitted_exclusively_on_training"] is True
    assert manifest["data_integrity_verification"]["validation_or_test_used_in_fitting_or_early_stopping"] is False
    assert manifest["data_integrity_verification"]["predictor_count"] == 13
    assert "DEPTH" not in manifest["data_integrity_verification"]["predictor_columns"]


def test_predictions_and_excluded_transition_zone():
    """Verify predictions file has 1,811 rows and excludes invalid targets from residuals."""
    pred_csv = BASE_DIR / "outputs" / "metrics" / "model_predictions_1811.csv"
    assert pred_csv.exists()
    df_pred = pd.read_csv(pred_csv)
    assert len(df_pred) == 1811

    invalid_rows = df_pred[~df_pred["IS_TARGET_VALID"]]
    assert len(invalid_rows) == 397

    for key in ["MLR", "RANDOM_FOREST", "CATBOOST", "GRADIENT_BOOSTING", "MLP"]:
        # Predictions exist across the entire borehole
        assert df_pred[f"PRED_{key}_BAR"].notna().all()
        # Residuals are strictly NaN on the 397 invalid transition-zone rows
        assert invalid_rows[f"RESIDUAL_{key}_BAR"].isna().all()
        # Residuals are finite on all 1,414 valid target rows
        valid_rows = df_pred[df_pred["IS_TARGET_VALID"]]
        assert valid_rows[f"RESIDUAL_{key}_BAR"].notna().all()


def test_evaluation_tables_and_provenance_separation():
    """Verify partition, provenance, formation, station, and feature importance tables."""
    metrics_dir = BASE_DIR / "outputs" / "metrics"

    # 1. Partition metrics
    part_df = pd.read_csv(metrics_dir / "partition_metrics.csv")
    subsets = set(part_df["partition_subset"].unique())
    assert "Training (Valid)" in subsets
    assert "Validation (Valid)" in subsets
    assert "Testing (Measured Anchors Only: Tests 20 & 21)" in subsets
    assert "Testing (Provisional Skagerrak Assumption Only)" in subsets

    # 2. Provenance metrics
    prov_df = pd.read_csv(metrics_dir / "provenance_metrics.csv")
    provs = set(prov_df["target_provenance"].unique())
    assert provs == {"measured_anchor", "physics_derived", "provisional_assumption"}

    # 3. Station evaluation and benchmarks
    st_df = pd.read_csv(metrics_dir / "station_evaluation_metrics.csv")
    assert len(st_df) == 21

    # Check Test 16 (Validation checkpoint)
    t16 = st_df[st_df["test_no"] == 16].iloc[0]
    assert t16["partition"] == "Validation"
    assert pytest.approx(t16["reconciled_station_p_bar"], abs=1e-3) == 341.713

    # Check Tests 20 and 21 (Independent Test stations)
    t20 = st_df[st_df["test_no"] == 20].iloc[0]
    t21 = st_df[st_df["test_no"] == 21].iloc[0]
    assert pytest.approx(t20["reconciled_station_p_bar"], abs=1e-3) == 368.706
    assert pytest.approx(t21["reconciled_station_p_bar"], abs=1e-3) == 368.818

    # Check Test 19 (Excluded alternative memory observation)
    t19 = st_df[st_df["test_no"] == 19].iloc[0]
    assert t19["station_provenance"] == "uncalibrated_memory_conflict"
    assert pytest.approx(t19["reconciled_station_p_bar"], abs=1e-3) == 357.714
    assert t19["target_source_test_no"] == 21

    # 4. Feature importance
    feat_df = pd.read_csv(metrics_dir / "feature_importance.csv")
    assert len(feat_df) == 5 * 13


def test_diagnostic_figures_generated():
    """Verify all 5 diagnostic plots exist and are non-empty."""
    fig_dir = BASE_DIR / "outputs" / "figures"
    expected_figs = [
        "actual_vs_predicted.png",
        "residuals_vs_depth_and_dist.png",
        "borehole_pressure_profiles.png",
        "feature_importance_comparison.png",
        "learning_curves.png",
    ]
    for fname in expected_figs:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Missing diagnostic figure: {fpath}"
        assert fpath.stat().st_size > 10_000

