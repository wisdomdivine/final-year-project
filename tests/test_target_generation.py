"""Unit tests for pore-pressure target generation and provenance integrity."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data.targets import generate_target_vectors, audit_target_generation


@pytest.fixture(scope="module")
def targets_df():
    """Load or generate the 1,811-observation target DataFrame."""
    target_csv = Path("dataset/F12_targets_1811.csv")
    if target_csv.exists():
        return pd.read_csv(target_csv)
    return generate_target_vectors()


def test_target_shape_and_columns(targets_df):
    """Verify exact 1,811-observation shape and required metadata columns."""
    assert len(targets_df) == 1811
    expected_cols = [
        "observation_index",
        "DEPTH",
        "TVD_MSL",
        "TVD_RKB",
        "FORMATION",
        "PARTITION",
        "IS_TARGET_VALID",
        "TARGET_PROVENANCE",
        "PP_TARGET_BAR",
        "PP_TARGET_UNCERTAINTY_BAR",
        "MEASURED_TEST_NO",
        "MEASURED_PRESSURE_BAR",
        "DERIVATION_METHOD",
        "PP_SENSITIVITY_RAMP_BAR",
        "PP_SENSITIVITY_TOPSEAL_BAR",
        "PP_SENSITIVITY_BASALSEAL_BAR",
    ]
    for col in expected_cols:
        assert col in targets_df.columns, f"Missing required column: {col}"


def test_provenance_and_validity_counts(targets_df):
    """Verify exact counts of valid targets and provenance classifications."""
    assert targets_df["IS_TARGET_VALID"].sum() == 1414
    assert (~targets_df["IS_TARGET_VALID"]).sum() == 397

    prov_counts = targets_df["TARGET_PROVENANCE"].value_counts().to_dict()
    assert prov_counts["measured_anchor"] == 17
    assert prov_counts["physics_derived"] == 1287
    assert prov_counts["provisional_assumption"] == 110
    assert prov_counts["invalid_unsupported"] == 397


def test_transition_zone_exclusion(targets_df):
    """Verify 3,326.0 to 3,386.5 m MD interval is completely excluded from primary targets."""
    trans_mask = (targets_df["DEPTH"] > 3326.0) & (targets_df["DEPTH"] <= 3386.5)
    trans = targets_df[trans_mask]

    assert len(trans) == 397
    assert np.all(~trans["IS_TARGET_VALID"])
    assert trans["PP_TARGET_BAR"].isna().all()
    assert (trans["TARGET_PROVENANCE"] == "invalid_unsupported").all()

    # Sensitivity targets must be fully populated and finite across the transition zone
    assert trans["PP_SENSITIVITY_RAMP_BAR"].notna().all()
    assert trans["PP_SENSITIVITY_TOPSEAL_BAR"].notna().all()
    assert trans["PP_SENSITIVITY_BASALSEAL_BAR"].notna().all()


def test_measured_anchors_fidelity(targets_df):
    """Verify that all 17 discrete measured anchors match Schlumberger pretests exactly."""
    anchors = targets_df[targets_df["TARGET_PROVENANCE"] == "measured_anchor"]
    assert len(anchors) == 17

    # Primary target must equal measured pressure at anchor nodes
    diff = (anchors["PP_TARGET_BAR"] - anchors["MEASURED_PRESSURE_BAR"]).abs()
    assert np.all(diff < 1e-5)

    # Tests 20 and 21 must be separate anchors
    t20_row = targets_df[targets_df["MEASURED_TEST_NO"] == 20].iloc[0]
    t21_row = targets_df[targets_df["MEASURED_TEST_NO"] == 21].iloc[0]

    assert pytest.approx(t20_row["PP_TARGET_BAR"], abs=1e-3) == 368.706
    assert pytest.approx(t21_row["PP_TARGET_BAR"], abs=1e-3) == 368.818
    assert t20_row["observation_index"] != t21_row["observation_index"]


def test_anti_leakage_split_isolation(targets_df):
    """Verify strict partition boundaries and zero leakage of test anchors."""
    train_part = targets_df[targets_df["PARTITION"] == "Training"]
    val_part = targets_df[targets_df["PARTITION"] == "Validation"]
    test_part = targets_df[targets_df["PARTITION"] == "Testing"]

    assert len(train_part) == 1268
    assert len(val_part) == 272
    assert len(test_part) == 271

    # Tests 20 & 21 must reside exclusively in Testing
    test_anchors = test_part["MEASURED_TEST_NO"].dropna().astype(int).tolist()
    assert 20 in test_anchors
    assert 21 in test_anchors

    train_anchors = train_part["MEASURED_TEST_NO"].dropna().astype(int).tolist()
    assert 20 not in train_anchors
    assert 21 not in train_anchors

    val_anchors = val_part["MEASURED_TEST_NO"].dropna().astype(int).tolist()
    assert 20 not in val_anchors
    assert 21 not in val_anchors
    assert 16 in val_anchors  # Test 16 is held out as intermediate validation checkpoint


def test_provisional_skagerrak_assumption(targets_df):
    """Verify Skagerrak non-anchor rows use provisional assumption, not measured ground truth."""
    skag_sub = targets_df[(targets_df["FORMATION"] == "Skagerrak") & (targets_df["TARGET_PROVENANCE"] != "measured_anchor")]
    assert len(skag_sub) == 110
    assert (skag_sub["TARGET_PROVENANCE"] == "provisional_assumption").all()
    assert np.allclose(skag_sub["PP_TARGET_BAR"].values, 368.762, atol=1e-3)


def test_target_splits_alignment():
    """Verify alignment of processed target files with predictor feature splits."""
    y_train = pd.read_csv("dataset/processed/y_train.csv")
    y_val = pd.read_csv("dataset/processed/y_val.csv")
    y_test = pd.read_csv("dataset/processed/y_test.csv")

    X_train = pd.read_csv("dataset/processed/X_train_raw.csv")
    X_val = pd.read_csv("dataset/processed/X_val_raw.csv")
    X_test = pd.read_csv("dataset/processed/X_test_raw.csv")

    assert len(y_train) == len(X_train) == 1268
    assert len(y_val) == len(X_val) == 272
    assert len(y_test) == len(X_test) == 271

    assert np.allclose(y_train["DEPTH"].values, X_train["DEPTH"].values)
    assert np.allclose(y_val["DEPTH"].values, X_val["DEPTH"].values)
    assert np.allclose(y_test["DEPTH"].values, X_test["DEPTH"].values)
