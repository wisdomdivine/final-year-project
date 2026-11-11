"""Unit tests for dataset integrity, depth continuity, and partitioning."""

import numpy as np
import pytest

from src.data.loader import load_predictor_dataset
from src.data.validation import validate_predictor_dataset, EXPECTED_PREDICTORS, DEPTH_COL
from src.data.preprocessing import split_dataset_by_depth, scale_dataset_splits


@pytest.fixture
def predictor_df():
    """Fixture providing raw 1,811 predictor dataset."""
    return load_predictor_dataset()


def test_predictor_shape_and_columns(predictor_df):
    """Test that dataset contains exactly 1,811 rows and 14 expected columns."""
    assert len(predictor_df) == 1811
    assert DEPTH_COL in predictor_df.columns
    for pred in EXPECTED_PREDICTORS:
        assert pred in predictor_df.columns


def test_predictor_completeness_and_finite(predictor_df):
    """Test that dataset has zero nulls and zero infinite values."""
    assert predictor_df.isnull().sum().sum() == 0
    assert np.isinf(predictor_df[EXPECTED_PREDICTORS].values).sum() == 0


def test_depth_monotonicity_and_step(predictor_df):
    """Test that DEPTH is strictly monotonically increasing with ~0.1524 m step."""
    depths = predictor_df[DEPTH_COL].values
    diffs = np.diff(depths)

    assert np.all(diffs > 0), "Depths must be strictly increasing"
    assert pytest.approx(float(np.median(diffs)), rel=1e-3) == 0.1524
    assert pytest.approx(float(depths[0]), abs=1e-3) == 3127.7052
    assert pytest.approx(float(depths[-1]), abs=1e-3) == 3403.5492


def test_validation_function(predictor_df):
    """Test comprehensive validation runner outputs PASS."""
    audit = validate_predictor_dataset(predictor_df)
    assert audit["status"] == "PASS"
    assert len(audit["failures"]) == 0
    assert audit["checks"]["row_count"]["pass"] is True
    assert audit["checks"]["missing_values"]["pass"] is True
    assert audit["checks"]["depth_continuity"]["pass"] is True


def test_depth_partitioning_counts_and_boundaries(predictor_df):
    """Test strictly contiguous 70:15:15 depth partitioning."""
    df_train, df_val, df_test = split_dataset_by_depth(predictor_df, depth_col=DEPTH_COL)

    assert len(df_train) == 1268
    assert len(df_val) == 272
    assert len(df_test) == 271
    assert len(df_train) + len(df_val) + len(df_test) == 1811

    # Check boundaries
    assert df_train[DEPTH_COL].min() == pytest.approx(3127.7052, abs=1e-4)
    assert df_train[DEPTH_COL].max() == pytest.approx(3320.7960, abs=1e-4)

    assert df_val[DEPTH_COL].min() == pytest.approx(3320.9484, abs=1e-4)
    assert df_val[DEPTH_COL].max() == pytest.approx(3362.2488, abs=1e-4)

    assert df_test[DEPTH_COL].min() == pytest.approx(3362.4012, abs=1e-4)
    assert df_test[DEPTH_COL].max() == pytest.approx(3403.5492, abs=1e-4)

    # Check non-overlap
    assert df_train[DEPTH_COL].max() < df_val[DEPTH_COL].min()
    assert df_val[DEPTH_COL].max() < df_test[DEPTH_COL].min()


def test_scaler_leakage_prevention(predictor_df):
    """Verify that scaler fitted on train standardizes train to mean ~ 0, std ~ 1."""
    df_train, df_val, df_test = split_dataset_by_depth(predictor_df, depth_col=DEPTH_COL)
    features = ["DT", "RHOB", "GR"]

    train_sc, val_sc, test_sc, scaler = scale_dataset_splits(
        df_train, df_val, df_test, feature_columns=features, scaler_type="standard"
    )

    for feat in features:
        train_mean = float(train_sc[feat].mean())
        train_std = float(train_sc[feat].std(ddof=0))
        assert pytest.approx(train_mean, abs=1e-5) == 0.0
        assert pytest.approx(train_std, abs=1e-4) == 1.0


def test_phase2_audit_deliverables():
    """Verify Phase 2 pre-execution audit tables exist and meet integrity criteria."""
    import pandas as pd
    from pathlib import Path

    audit_dir = Path("outputs/audit")

    # 1. Trajectory reconciliation
    traj_path = audit_dir / "trajectory_reconciliation.csv"
    assert traj_path.exists()
    df_traj = pd.read_csv(traj_path)
    assert len(df_traj) == 1811
    assert (df_traj["delta_TVD_MSL_m"] >= 33.8).all()
    assert (df_traj["delta_TVD_MSL_m"] <= 34.5).all()

    # 2. Pressure station reconciliation
    press_path = audit_dir / "pressure_station_reconciliation.csv"
    assert press_path.exists()
    df_press = pd.read_csv(press_path)
    assert len(df_press) == 24
    for t in [18, 19, 20, 21]:
        assert t in df_press["test_no"].values

    # 3. Overburden sensitivity
    ovb_path = audit_dir / "overburden_sensitivity.csv"
    assert ovb_path.exists()
    df_ovb = pd.read_csv(ovb_path)
    assert len(df_ovb) == 1811
    assert (df_ovb["Sv_traj2_mid_1.88_bar"].diff().dropna() > 0).all()

