"""
Comprehensive Model Training, Evaluation, and Diagnostics Pipeline
for Well 15/9-F-12 (Volve Field) Pore-Pressure Prediction.

Enforces strict anti-leakage rules:
1. Contiguous 70:15:15 depth split (Train: 1,268; Val: 272; Test: 271).
2. Feature transformations (log10 on K, RS, RD, RT) and StandardScaler fitted
   exclusively on Training data.
3. Invalid targets (MD 3,326.0 to 3,386.5 m, n=397) excluded from all training
   and primary metric calculations.
4. Predictor matrix strictly restricted to the 13 petrophysical and drilling
   logs (DT, GR, NPHI, RD, RHOB, RS, RT, ROP, BVW, K, PHIF, SW, VCL).
   DEPTH, TVD_MSL, TVD_RKB, TARGET_PROVENANCE, MEASURED_TEST_NO, and
   MEASURED_PRESSURE_BAR are never used as predictors.
5. Separate reporting by partition, target provenance, formation, and discrete
   Schlumberger pretest stations (Tests 1-16, 19, 20, 21).
"""

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import catboost
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import yaml

from src.data.validation import EXPECTED_PREDICTORS, DEPTH_COL
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.residuals import compute_residuals, analyze_residuals
from src.models import (
    MultipleLinearRegressionModel,
    RandomForestModel,
    CatBoostModel,
    GradientBoostingModel,
    MLPModel,
)
from src.physics.hydrostatic import calculate_hydrostatic_pressure

matplotlib.use("Agg")

RANDOM_SEED = 42
LOG_COLS = ["K", "RS", "RD", "RT"]

FORBIDDEN_PREDICTOR_COLS = {
    "DEPTH",
    "TVD_MSL",
    "TVD_RKB",
    "observation_index",
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
}


def _save_dual(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    """Save a DataFrame to both CSV and JSON formats."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    records = json.loads(df.to_json(orient="records"))
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)


def prepare_training_data(
    base_dir: Path,
) -> Dict[str, Any]:
    """
    Load predictor dataset and target vectors, verify alignment and anti-leakage
    constraints, and construct raw and log-scaled feature matrices fitted
    strictly on Training data.
    """
    pred_path = base_dir / "dataset" / "F12_predictor_dataset_1811.csv"
    targ_path = base_dir / "dataset" / "F12_targets_1811.csv"

    df_pred = pd.read_csv(pred_path)
    df_targ = pd.read_csv(targ_path)

    if len(df_pred) != 1811 or len(df_targ) != 1811:
        raise ValueError("Predictor and target datasets must contain exactly 1,811 rows.")

    if not np.allclose(df_pred[DEPTH_COL].values, df_targ[DEPTH_COL].values, atol=1e-5):
        raise ValueError("Predictor and target DEPTH vectors are not aligned.")

    predictor_cols = list(EXPECTED_PREDICTORS)
    if len(predictor_cols) != 13:
        raise ValueError(f"Expected 13 predictors, got {len(predictor_cols)}")

    overlap = set(predictor_cols).intersection(FORBIDDEN_PREDICTOR_COLS)
    if overlap:
        raise ValueError(f"Forbidden leakage columns found in predictor list: {overlap}")

    # Contiguous 70:15:15 depth split masks
    train_mask = (df_targ["PARTITION"] == "Training").values
    val_mask = (df_targ["PARTITION"] == "Validation").values
    test_mask = (df_targ["PARTITION"] == "Testing").values

    if int(train_mask.sum()) != 1268 or int(val_mask.sum()) != 272 or int(test_mask.sum()) != 271:
        raise ValueError("Partition counts do not match 1268:272:271 split.")

    valid_mask = df_targ["IS_TARGET_VALID"].astype(bool).values

    # Raw 13-feature matrices (for tree/boosting models)
    X_all_raw = df_pred[predictor_cols].copy()
    X_train_raw = X_all_raw.iloc[train_mask].copy().reset_index(drop=True)
    X_val_raw = X_all_raw.iloc[val_mask].copy().reset_index(drop=True)
    X_test_raw = X_all_raw.iloc[test_mask].copy().reset_index(drop=True)

    # Log10 + Standard-scaled 13-feature matrices (fitted strictly on X_train)
    X_all_log = X_all_raw.copy()
    for col in LOG_COLS:
        X_all_log[col] = np.log10(np.clip(X_all_log[col].values, a_min=1e-6, a_max=None))

    X_train_log = X_all_log.iloc[train_mask].copy()
    scaler_log = StandardScaler()
    scaler_log.fit(X_train_log[predictor_cols])

    X_all_log_scaled = pd.DataFrame(
        scaler_log.transform(X_all_log[predictor_cols]),
        columns=predictor_cols,
        index=X_all_raw.index,
    )
    X_train_log_scaled = X_all_log_scaled.iloc[train_mask].copy().reset_index(drop=True)
    X_val_log_scaled = X_all_log_scaled.iloc[val_mask].copy().reset_index(drop=True)
    X_test_log_scaled = X_all_log_scaled.iloc[test_mask].copy().reset_index(drop=True)

    # Also construct Standard-scaled without log10 (fitted strictly on X_train) for ablation audit
    scaler_std = StandardScaler()
    scaler_std.fit(X_train_raw[predictor_cols])
    X_all_std_scaled = pd.DataFrame(
        scaler_std.transform(X_all_raw[predictor_cols]),
        columns=predictor_cols,
        index=X_all_raw.index,
    )

    y_all = df_targ["PP_TARGET_BAR"].values.astype(float)
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    train_valid_mask = valid_mask[train_mask]
    val_valid_mask = valid_mask[val_mask]
    test_valid_mask = valid_mask[test_mask]

    if not np.all(train_valid_mask):
        raise ValueError("All 1,268 training targets must be valid.")
    if int(val_valid_mask.sum()) != 34:
        raise ValueError(f"Expected 34 valid validation targets, found {val_valid_mask.sum()}")
    if int(test_valid_mask.sum()) != 112:
        raise ValueError(f"Expected 112 valid testing targets, found {test_valid_mask.sum()}")

    return {
        "df_pred": df_pred,
        "df_targ": df_targ,
        "predictor_cols": predictor_cols,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "valid_mask": valid_mask,
        "train_valid_mask": train_valid_mask,
        "val_valid_mask": val_valid_mask,
        "test_valid_mask": test_valid_mask,
        "X_all_raw": X_all_raw,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "X_all_log_scaled": X_all_log_scaled,
        "X_train_log_scaled": X_train_log_scaled,
        "X_val_log_scaled": X_val_log_scaled,
        "X_test_log_scaled": X_test_log_scaled,
        "X_all_std_scaled": X_all_std_scaled,
        "y_all": y_all,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler_log": scaler_log,
        "scaler_std": scaler_std,
    }


def train_all_models(
    data_bundle: Dict[str, Any],
    models_config_path: Path,
    models_out_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Instantiate and train the 5 candidate regression models strictly on the
    1,268 valid Training observations. Save model artifacts and manifest.
    """
    with open(models_config_path, "r") as f:
        cfg = yaml.safe_load(f)["models"]

    models_out_dir.mkdir(parents=True, exist_ok=True)

    X_train_raw = data_bundle["X_train_raw"]
    X_train_log_scaled = data_bundle["X_train_log_scaled"]
    X_all_raw = data_bundle["X_all_raw"]
    X_all_log_scaled = data_bundle["X_all_log_scaled"]
    y_train = data_bundle["y_train"]
    train_valid_mask = data_bundle["train_valid_mask"]

    # Strictly filter training data by valid mask (all 1,268 rows are valid)
    X_tr_raw_valid = X_train_raw.iloc[train_valid_mask]
    X_tr_log_valid = X_train_log_scaled.iloc[train_valid_mask]
    y_tr_valid = y_train[train_valid_mask]

    model_specs = [
        {
            "key": "mlr",
            "display_name": "Multiple Linear Regression",
            "short_name": "MLR",
            "family": "linear",
            "feature_set": "log10_standard_scaled_13",
            "cls": MultipleLinearRegressionModel,
            "params": cfg.get("mlr", {}),
        },
        {
            "key": "random_forest",
            "display_name": "Random Forest Regressor",
            "short_name": "Random Forest",
            "family": "tree_bagging",
            "feature_set": "raw_unscaled_13",
            "cls": RandomForestModel,
            "params": cfg.get("random_forest", {}),
        },
        {
            "key": "catboost",
            "display_name": "CatBoost Regressor",
            "short_name": "CatBoost",
            "family": "gradient_boosting",
            "feature_set": "raw_unscaled_13",
            "cls": CatBoostModel,
            "params": cfg.get("catboost", {}),
        },
        {
            "key": "gradient_boosting",
            "display_name": "Gradient Boosting Regressor",
            "short_name": "Gradient Boosting",
            "family": "gradient_boosting",
            "feature_set": "raw_unscaled_13",
            "cls": GradientBoostingModel,
            "params": cfg.get("gradient_boosting", {}),
        },
        {
            "key": "mlp",
            "display_name": "Multi-Layer Perceptron",
            "short_name": "MLP",
            "family": "neural_network",
            "feature_set": "log10_standard_scaled_13",
            "cls": MLPModel,
            "params": cfg.get("mlp", {}),
        },
    ]

    trained_registry: Dict[str, Dict[str, Any]] = {}

    for spec in model_specs:
        key = spec["key"]
        model_instance = spec["cls"](params=spec["params"])

        if spec["feature_set"] == "log10_standard_scaled_13":
            X_fit = X_tr_log_valid
            X_pred_all = X_all_log_scaled
        else:
            X_fit = X_tr_raw_valid
            X_pred_all = X_all_raw

        # Fit strictly on training data (never pass validation or test sets)
        model_instance.fit(X_fit, y_tr_valid)
        y_pred_all = model_instance.predict(X_pred_all)

        artifact_path = models_out_dir / f"{key}_model.joblib"
        model_instance.save(artifact_path)

        trained_registry[key] = {
            "key": key,
            "display_name": spec["display_name"],
            "short_name": spec["short_name"],
            "family": spec["family"],
            "feature_set": spec["feature_set"],
            "model": model_instance,
            "params": model_instance.params,
            "artifact_path": str(artifact_path),
            "y_pred_all": y_pred_all,
        }

    # Save fitted scalers
    scaler_log_path = models_out_dir / "log_standard_scaler.joblib"
    joblib.dump(data_bundle["scaler_log"], scaler_log_path)

    # Save training manifest with environment & reproducibility details
    manifest = {
        "baseline_commit": "d0ea70d",
        "random_seed": RANDOM_SEED,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "data_integrity_verification": {
            "total_observations": 1811,
            "training_observations_total": 1268,
            "training_observations_valid": 1268,
            "validation_observations_total": 272,
            "validation_observations_valid": 34,
            "validation_observations_excluded_transition": 238,
            "testing_observations_total": 271,
            "testing_observations_valid": 112,
            "testing_observations_excluded_transition": 159,
            "predictor_count": len(data_bundle["predictor_cols"]),
            "predictor_columns": data_bundle["predictor_cols"],
            "log10_transformed_columns": LOG_COLS,
            "forbidden_columns_excluded": sorted(list(FORBIDDEN_PREDICTOR_COLS)),
            "scaler_fitted_exclusively_on_training": True,
            "validation_or_test_used_in_fitting_or_early_stopping": False,
        },
        "models": {
            k: {
                "display_name": v["display_name"],
                "short_name": v["short_name"],
                "family": v["family"],
                "feature_set": v["feature_set"],
                "hyperparameters": {
                    pk: (list(pv) if isinstance(pv, tuple) else pv)
                    for pk, pv in v["params"].items()
                },
                "artifact_path": v["artifact_path"],
            }
            for k, v in trained_registry.items()
        },
    }

    manifest_path = models_out_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return trained_registry


def evaluate_all_models(
    data_bundle: Dict[str, Any],
    trained_registry: Dict[str, Dict[str, Any]],
    metrics_dir: Path,
    tables_dir: Path,
    audit_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Compute comprehensive evaluation metrics across partitions, provenances,
    formations, discrete physical stations, residual distributions, and feature
    importances.
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_targ = data_bundle["df_targ"]
    y_all = data_bundle["y_all"]
    valid_mask = data_bundle["valid_mask"]
    train_mask = data_bundle["train_mask"]
    val_mask = data_bundle["val_mask"]
    test_mask = data_bundle["test_mask"]
    predictor_cols = data_bundle["predictor_cols"]

    # 1. Master 1,811-row Predictions Table
    pred_df = df_targ[
        [
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
        ]
    ].copy()

    # Add physical hydrostatic and overburden baselines for plotting/context (Trajectory 1)
    pred_df["PH_HYDROSTATIC_BAR"] = np.round(
        calculate_hydrostatic_pressure(pred_df["TVD_MSL"].values), 4
    )
    ovb_df = pd.read_csv(audit_dir / "overburden_sensitivity.csv")
    pred_df["SV_OVERBURDEN_BAR"] = np.round(ovb_df["Sv_traj1_rho1.9_bar"].values, 4)

    for key, info in trained_registry.items():
        col_pred = f"PRED_{key.upper()}_BAR"
        col_res = f"RESIDUAL_{key.upper()}_BAR"
        yp = info["y_pred_all"]
        pred_df[col_pred] = np.round(yp, 4)
        res = np.where(valid_mask, y_all - yp, np.nan)
        pred_df[col_res] = np.round(res, 4)

    _save_dual(
        pred_df,
        metrics_dir / "model_predictions_1811.csv",
        metrics_dir / "model_predictions_1811.json",
    )
    _save_dual(
        pred_df,
        tables_dir / "model_predictions_1811.csv",
        tables_dir / "model_predictions_1811.json",
    )

    # Save per-partition prediction CSVs
    pred_df[pred_df["PARTITION"] == "Training"].to_csv(
        metrics_dir / "predictions_train.csv", index=False
    )
    pred_df[pred_df["PARTITION"] == "Validation"].to_csv(
        metrics_dir / "predictions_val.csv", index=False
    )
    pred_df[pred_df["PARTITION"] == "Testing"].to_csv(
        metrics_dir / "predictions_test.csv", index=False
    )

    # 2. Partition Metrics (Valid targets only, plus breakdown of Test measured vs provisional)
    partition_subsets = [
        ("Training (Valid)", train_mask & valid_mask, "primary_calibration_and_derived"),
        ("Validation (Valid)", val_mask & valid_mask, "lower_sleipner_valid_interval"),
        ("Testing (All Valid: 2 Anchors + 110 Provisional)", test_mask & valid_mask, "mixed_2_measured_110_provisional_not_independent_physical_validation"),
        ("Testing (Measured Anchors Only: Tests 20 & 21)", test_mask & (df_targ["TARGET_PROVENANCE"].values == "measured_anchor"), "independent_physical_test_stations"),
        ("Testing (Provisional Skagerrak Assumption Only)", test_mask & (df_targ["TARGET_PROVENANCE"].values == "provisional_assumption"), "provisional_constant_assumption_368.762_bar"),
        ("Full Dataset (All Valid Targets)", valid_mask, "all_1414_valid_targets"),
        ("Full Dataset (Measured Anchors + Physics-Derived Only)", (df_targ["TARGET_PROVENANCE"].isin(["measured_anchor", "physics_derived"])).values, "excludes_110_provisional_skagerrak_targets"),
    ]

    partition_rows = []
    for key, info in trained_registry.items():
        yp = info["y_pred_all"]
        for subset_label, mask, note in partition_subsets:
            m = compute_regression_metrics(y_all[mask], yp[mask])
            res_stats = analyze_residuals(y_all[mask] - yp[mask])
            partition_rows.append(
                {
                    "model_key": key,
                    "model_name": info["short_name"],
                    "partition_subset": subset_label,
                    "n_valid": int(mask.sum()),
                    "MAE_bar": round(m["MAE"], 4),
                    "RMSE_bar": round(m["RMSE"], 4),
                    "MSE_bar2": round(m["MSE"], 4),
                    "R2": round(m["R2"], 4),
                    "MAPE_pct": round(m["MAPE"], 4),
                    "MaxAE_bar": round(m["MaxAE"], 4),
                    "mean_residual_bar": round(res_stats.get("mean_residual", np.nan), 4),
                    "std_residual_bar": round(res_stats.get("std_residual", np.nan), 4),
                    "median_residual_bar": round(res_stats.get("median_residual", np.nan), 4),
                    "evaluation_note": note,
                }
            )

    df_partition_metrics = pd.DataFrame(partition_rows)
    _save_dual(
        df_partition_metrics,
        metrics_dir / "partition_metrics.csv",
        metrics_dir / "partition_metrics.json",
    )
    _save_dual(
        df_partition_metrics,
        tables_dir / "partition_metrics.csv",
        tables_dir / "partition_metrics.json",
    )

    # 3. Provenance Metrics Breakdown
    prov_col = df_targ["TARGET_PROVENANCE"].values
    provenance_subsets = [
        ("measured_anchor", "All Partitions", prov_col == "measured_anchor", "All 17 primary measured pretest anchors"),
        ("measured_anchor", "Training", (prov_col == "measured_anchor") & train_mask, "14 training anchors (Tests 1-12, 14, 15)"),
        ("measured_anchor", "Validation", (prov_col == "measured_anchor") & val_mask, "1 independent validation anchor (Test 16)"),
        ("measured_anchor", "Testing", (prov_col == "measured_anchor") & test_mask, "2 independent test anchors (Tests 20 & 21)"),
        ("physics_derived", "All Partitions", prov_col == "physics_derived", "All 1,287 regime-equation/interpolated targets"),
        ("physics_derived", "Training", (prov_col == "physics_derived") & train_mask, "1,254 Hugin + upper/mid Sleipner derived targets"),
        ("physics_derived", "Validation", (prov_col == "physics_derived") & val_mask, "33 lower Sleipner piece-wise interpolated targets"),
        ("provisional_assumption", "Testing", prov_col == "provisional_assumption", "110 provisional constant-compartment Skagerrak rows (not independent physical ground truth)"),
    ]

    prov_rows = []
    for key, info in trained_registry.items():
        yp = info["y_pred_all"]
        for prov_name, part_label, mask, desc in provenance_subsets:
            m = compute_regression_metrics(y_all[mask], yp[mask])
            res = y_all[mask] - yp[mask]
            prov_rows.append(
                {
                    "model_key": key,
                    "model_name": info["short_name"],
                    "target_provenance": prov_name,
                    "partition": part_label,
                    "n_observations": int(mask.sum()),
                    "MAE_bar": round(m["MAE"], 4),
                    "RMSE_bar": round(m["RMSE"], 4),
                    "R2": round(m["R2"], 4) if len(res) > 1 and np.std(y_all[mask]) > 1e-9 else np.nan,
                    "MAPE_pct": round(m["MAPE"], 4),
                    "MaxAE_bar": round(m["MaxAE"], 4),
                    "mean_residual_bar": round(float(np.mean(res)), 4),
                    "description": desc,
                }
            )

    df_prov_metrics = pd.DataFrame(prov_rows)
    _save_dual(
        df_prov_metrics,
        metrics_dir / "provenance_metrics.csv",
        metrics_dir / "provenance_metrics.json",
    )
    _save_dual(
        df_prov_metrics,
        tables_dir / "provenance_metrics.csv",
        tables_dir / "provenance_metrics.json",
    )

    # 4. Formation Metrics Breakdown
    form_col = df_targ["FORMATION"].values
    formation_subsets = [
        ("Hugin", "Training", (form_col == "Hugin") & valid_mask),
        ("Sleipner", "Training", (form_col == "Sleipner") & train_mask & valid_mask),
        ("Sleipner", "Validation (Valid Lower Sleipner)", (form_col == "Sleipner") & val_mask & valid_mask),
        ("Sleipner", "All Valid Sleipner", (form_col == "Sleipner") & valid_mask),
        ("Skagerrak", "Testing (Measured Anchors: Tests 20 & 21)", (form_col == "Skagerrak") & (prov_col == "measured_anchor")),
        ("Skagerrak", "Testing (Provisional Assumption: n=110)", (form_col == "Skagerrak") & (prov_col == "provisional_assumption")),
        ("Skagerrak", "Testing (All 112 Rows)", (form_col == "Skagerrak") & valid_mask),
    ]

    form_rows = []
    for key, info in trained_registry.items():
        yp = info["y_pred_all"]
        for form_name, sub_label, mask in formation_subsets:
            m = compute_regression_metrics(y_all[mask], yp[mask])
            res = y_all[mask] - yp[mask]
            form_rows.append(
                {
                    "model_key": key,
                    "model_name": info["short_name"],
                    "formation": form_name,
                    "subset": sub_label,
                    "n_valid": int(mask.sum()),
                    "MAE_bar": round(m["MAE"], 4),
                    "RMSE_bar": round(m["RMSE"], 4),
                    "R2": round(m["R2"], 4) if len(res) > 1 and np.std(y_all[mask]) > 1e-9 else np.nan,
                    "MAPE_pct": round(m["MAPE"], 4),
                    "MaxAE_bar": round(m["MaxAE"], 4),
                    "mean_residual_bar": round(float(np.mean(res)), 4),
                    "mean_predicted_bar": round(float(np.mean(yp[mask])), 4),
                    "mean_target_bar": round(float(np.mean(y_all[mask])), 4),
                }
            )

    df_form_metrics = pd.DataFrame(form_rows)
    _save_dual(
        df_form_metrics,
        metrics_dir / "formation_metrics.csv",
        metrics_dir / "formation_metrics.json",
    )
    _save_dual(
        df_form_metrics,
        tables_dir / "formation_metrics.csv",
        tables_dir / "formation_metrics.json",
    )

    # 5. Discrete Physical Station Evaluation (All Run 10 Stations, highlighting Test 16, Tests 20 & 21, and Test 19)
    comp_df = pd.read_csv(audit_dir / "target_vs_measured_comparison.csv")
    station_rows = []

    for _, st in comp_df.iterrows():
        test_no = int(st["test_no"])
        grid_md = float(st["grid_md"])
        # Locate exact grid index
        idx = int(np.argmin(np.abs(df_targ["DEPTH"].values - grid_md)))
        reconciled_p = st["reconciled_station_p_bar"]
        has_valid_station_p = pd.notna(reconciled_p)

        row_dict: Dict[str, Any] = {
            "test_no": test_no,
            "station_md_m": float(st["station_md"]),
            "grid_md_m": float(df_targ.loc[idx, "DEPTH"]),
            "grid_index": idx,
            "formation": str(df_targ.loc[idx, "FORMATION"]),
            "partition": str(st["partition"]),
            "station_provenance": str(st["station_provenance"]),
            "station_role": str(st["station_role"]),
            "reconciled_station_p_bar": float(reconciled_p) if has_valid_station_p else np.nan,
            "conflicting_table_p_bar": float(st["conflicting_table_p_bar"]) if pd.notna(st["conflicting_table_p_bar"]) else np.nan,
            "grid_target_p_bar": float(st["grid_target_bar"]) if pd.notna(st["grid_target_bar"]) else np.nan,
            "target_source_test_no": int(st["target_source_test_no"]) if pd.notna(st["target_source_test_no"]) else np.nan,
        }

        for key, info in trained_registry.items():
            pred_val = float(info["y_pred_all"][idx])
            row_dict[f"pred_{key}_bar"] = round(pred_val, 4)
            if has_valid_station_p:
                # Residual against actual physical station pressure (station_p - pred_p)
                err = float(reconciled_p) - pred_val
                row_dict[f"error_{key}_bar"] = round(err, 4)
                row_dict[f"abs_error_{key}_bar"] = round(abs(err), 4)
            else:
                row_dict[f"error_{key}_bar"] = np.nan
                row_dict[f"abs_error_{key}_bar"] = np.nan

        station_rows.append(row_dict)

    df_station_eval = pd.DataFrame(station_rows)
    _save_dual(
        df_station_eval,
        metrics_dir / "station_evaluation_metrics.csv",
        metrics_dir / "station_evaluation_metrics.json",
    )
    _save_dual(
        df_station_eval,
        tables_dir / "station_evaluation_metrics.csv",
        tables_dir / "station_evaluation_metrics.json",
    )

    # Also build a tidy summary table specifically for the key benchmark station groups:
    # (a) 15 Training stations (Tests 1-15, using reconciled 334.161 bar for Test 13)
    # (b) Independent Validation Checkpoint: Test 16 (341.713 bar)
    # (c) Independent Test Stations: Test 20 (368.706 bar), Test 21 (368.818 bar), and combined Tests 20 & 21
    # (d) Excluded Alternative Memory Station: Test 19 (357.714 bar)
    key_station_summary_rows = []
    train_st_mask = df_station_eval["test_no"].isin(list(range(1, 16)))
    t16_row = df_station_eval[df_station_eval["test_no"] == 16].iloc[0]
    t20_row = df_station_eval[df_station_eval["test_no"] == 20].iloc[0]
    t21_row = df_station_eval[df_station_eval["test_no"] == 21].iloc[0]
    t19_row = df_station_eval[df_station_eval["test_no"] == 19].iloc[0]

    for key, info in trained_registry.items():
        tr_true = df_station_eval.loc[train_st_mask, "reconciled_station_p_bar"].values
        tr_pred = df_station_eval.loc[train_st_mask, f"pred_{key}_bar"].values
        tr_m = compute_regression_metrics(tr_true, tr_pred)

        t20_21_true = np.array([t20_row["reconciled_station_p_bar"], t21_row["reconciled_station_p_bar"]])
        t20_21_pred = np.array([t20_row[f"pred_{key}_bar"], t21_row[f"pred_{key}_bar"]])
        t20_21_m = compute_regression_metrics(t20_21_true, t20_21_pred)

        all_18_true = np.concatenate([tr_true, [t16_row["reconciled_station_p_bar"]], t20_21_true])
        all_18_pred = np.concatenate([tr_pred, [t16_row[f"pred_{key}_bar"]], t20_21_pred])
        all_18_m = compute_regression_metrics(all_18_true, all_18_pred)

        key_station_summary_rows.append(
            {
                "model_key": key,
                "model_name": info["short_name"],
                "train_15_stations_MAE_bar": round(tr_m["MAE"], 4),
                "train_15_stations_RMSE_bar": round(tr_m["RMSE"], 4),
                "train_15_stations_R2": round(tr_m["R2"], 4),
                "val_test16_measured_bar": round(float(t16_row["reconciled_station_p_bar"]), 3),
                "val_test16_pred_bar": round(float(t16_row[f"pred_{key}_bar"]), 4),
                "val_test16_residual_bar": round(float(t16_row[f"error_{key}_bar"]), 4),
                "val_test16_abs_error_bar": round(float(t16_row[f"abs_error_{key}_bar"]), 4),
                "test20_measured_bar": round(float(t20_row["reconciled_station_p_bar"]), 3),
                "test20_pred_bar": round(float(t20_row[f"pred_{key}_bar"]), 4),
                "test20_residual_bar": round(float(t20_row[f"error_{key}_bar"]), 4),
                "test21_measured_bar": round(float(t21_row["reconciled_station_p_bar"]), 3),
                "test21_pred_bar": round(float(t21_row[f"pred_{key}_bar"]), 4),
                "test21_residual_bar": round(float(t21_row[f"error_{key}_bar"]), 4),
                "test20_21_MAE_bar": round(t20_21_m["MAE"], 4),
                "test20_21_RMSE_bar": round(t20_21_m["RMSE"], 4),
                "all_18_calibrated_stations_MAE_bar": round(all_18_m["MAE"], 4),
                "all_18_calibrated_stations_RMSE_bar": round(all_18_m["RMSE"], 4),
                "all_18_calibrated_stations_R2": round(all_18_m["R2"], 4),
                "excluded_test19_memory_bar": round(float(t19_row["reconciled_station_p_bar"]), 3),
                "excluded_test19_pred_bar": round(float(t19_row[f"pred_{key}_bar"]), 4),
                "excluded_test19_diff_bar": round(float(t19_row[f"error_{key}_bar"]), 4),
            }
        )

    df_station_summary = pd.DataFrame(key_station_summary_rows)
    _save_dual(
        df_station_summary,
        metrics_dir / "physical_station_benchmarks.csv",
        metrics_dir / "physical_station_benchmarks.json",
    )
    _save_dual(
        df_station_summary,
        tables_dir / "physical_station_benchmarks.csv",
        tables_dir / "physical_station_benchmarks.json",
    )

    # 6. Feature Importance (Native + Model-Independent Permutation Importance)
    X_tr_raw = data_bundle["X_train_raw"]
    X_tr_log = data_bundle["X_train_log_scaled"]
    y_tr = data_bundle["y_train"]

    X_val_raw_valid = data_bundle["X_val_raw"].iloc[data_bundle["val_valid_mask"]]
    X_val_log_valid = data_bundle["X_val_log_scaled"].iloc[data_bundle["val_valid_mask"]]
    y_val_valid = data_bundle["y_val"][data_bundle["val_valid_mask"]]

    feat_rows = []
    for key, info in trained_registry.items():
        model_obj = info["model"]
        if info["feature_set"] == "log10_standard_scaled_13":
            X_perm_tr = X_tr_log
            X_perm_val = X_val_log_valid
        else:
            X_perm_tr = X_tr_raw
            X_perm_val = X_val_raw_valid

        # Native importance / coefficients
        if key == "mlr":
            native_raw = np.abs(model_obj.coef_)
            native_signed = model_obj.coef_
            native_type = "abs_standardized_coefficient"
        elif hasattr(model_obj, "feature_importances_"):
            native_raw = np.asarray(model_obj.feature_importances_, dtype=float)
            native_signed = native_raw
            native_type = "tree_split_or_gain_importance"
        else:
            native_raw = np.full(len(predictor_cols), np.nan)
            native_signed = np.full(len(predictor_cols), np.nan)
            native_type = "not_applicable"

        if np.all(np.isfinite(native_raw)) and np.sum(native_raw) > 0:
            native_norm_pct = (native_raw / np.sum(native_raw)) * 100.0
        else:
            native_norm_pct = np.full(len(predictor_cols), np.nan)

        # Model-independent permutation importance on Training and Validation sets
        # Using negative RMSE scoring via sklearn's neg_root_mean_squared_error
        perm_tr = permutation_importance(
            model_obj.model,
            X_perm_tr,
            y_tr,
            scoring="neg_root_mean_squared_error",
            n_repeats=15,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        perm_val = permutation_importance(
            model_obj.model,
            X_perm_val,
            y_val_valid,
            scoring="neg_root_mean_squared_error",
            n_repeats=15,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        perm_tr_mean = np.maximum(perm_tr.importances_mean, 0.0)
        perm_tr_pct = (
            (perm_tr_mean / np.sum(perm_tr_mean)) * 100.0
            if np.sum(perm_tr_mean) > 0
            else np.zeros_like(perm_tr_mean)
        )

        for i, col in enumerate(predictor_cols):
            feat_rows.append(
                {
                    "model_key": key,
                    "model_name": info["short_name"],
                    "feature": col,
                    "native_importance_type": native_type,
                    "native_importance_raw": round(float(native_raw[i]), 6) if np.isfinite(native_raw[i]) else np.nan,
                    "native_signed_value": round(float(native_signed[i]), 6) if np.isfinite(native_signed[i]) else np.nan,
                    "native_importance_pct": round(float(native_norm_pct[i]), 4) if np.isfinite(native_norm_pct[i]) else np.nan,
                    "perm_train_rmse_increase_bar": round(float(perm_tr.importances_mean[i]), 5),
                    "perm_train_rmse_std_bar": round(float(perm_tr.importances_std[i]), 5),
                    "perm_train_importance_pct": round(float(perm_tr_pct[i]), 4),
                    "perm_val_rmse_increase_bar": round(float(perm_val.importances_mean[i]), 5),
                    "perm_val_rmse_std_bar": round(float(perm_val.importances_std[i]), 5),
                }
            )

    df_feat_imp = pd.DataFrame(feat_rows)
    _save_dual(
        df_feat_imp,
        metrics_dir / "feature_importance.csv",
        metrics_dir / "feature_importance.json",
    )
    _save_dual(
        df_feat_imp,
        tables_dir / "feature_importance.csv",
        tables_dir / "feature_importance.json",
    )

    return {
        "predictions_1811": pred_df,
        "partition_metrics": df_partition_metrics,
        "provenance_metrics": df_prov_metrics,
        "formation_metrics": df_form_metrics,
        "station_evaluation": df_station_eval,
        "station_benchmarks": df_station_summary,
        "feature_importance": df_feat_imp,
    }


def generate_diagnostic_figures(
    data_bundle: Dict[str, Any],
    trained_registry: Dict[str, Dict[str, Any]],
    eval_tables: Dict[str, pd.DataFrame],
    figures_dir: Path,
) -> List[Path]:
    """
    Generate the 5 diagnostic figures required for model evaluation:
    1. actual_vs_predicted.png
    2. residuals_vs_depth_and_dist.png
    3. borehole_pressure_profiles.png
    4. feature_importance_comparison.png
    5. learning_curves.png
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    pred_df = eval_tables["predictions_1811"]
    valid_df = pred_df[pred_df["IS_TARGET_VALID"]].copy()
    station_df = eval_tables["station_evaluation"]
    feat_df = eval_tables["feature_importance"]

    model_keys = list(trained_registry.keys())
    model_colors = {
        "mlr": "#1f77b4",
        "random_forest": "#2ca02c",
        "catboost": "#d62728",
        "gradient_boosting": "#ff7f0e",
        "mlp": "#9467bd",
    }

    # -------------------------------------------------------------------------
    # Figure 1: Actual vs. Predicted Pore Pressure Parity Plots (5 panels + summary)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=200)
    axes_flat = axes.flatten()

    prov_styles = [
        ("Training: Physics-Derived (n=1254)", (valid_df["PARTITION"] == "Training") & (valid_df["TARGET_PROVENANCE"] == "physics_derived"), "#4c72b0", "o", 14, 0.35),
        ("Training: Measured Anchors (n=14)", (valid_df["PARTITION"] == "Training") & (valid_df["TARGET_PROVENANCE"] == "measured_anchor"), "#003366", "D", 42, 0.95),
        ("Validation: Lower Sleipner Derived (n=33)", (valid_df["PARTITION"] == "Validation") & (valid_df["TARGET_PROVENANCE"] == "physics_derived"), "#dd8452", "s", 22, 0.75),
        ("Validation: Test 16 Anchor (n=1)", (valid_df["PARTITION"] == "Validation") & (valid_df["TARGET_PROVENANCE"] == "measured_anchor"), "#e65100", "*", 130, 1.0),
        ("Testing: Provisional Skagerrak (n=110)", (valid_df["PARTITION"] == "Testing") & (valid_df["TARGET_PROVENANCE"] == "provisional_assumption"), "#c44e52", "^", 24, 0.45),
        ("Testing: Tests 20 & 21 Anchors (n=2)", (valid_df["PARTITION"] == "Testing") & (valid_df["TARGET_PROVENANCE"] == "measured_anchor"), "#8b0000", "P", 95, 1.0),
    ]

    for idx, key in enumerate(model_keys):
        ax = axes_flat[idx]
        info = trained_registry[key]
        col_pred = f"PRED_{key.upper()}_BAR"

        for label, mask, color, marker, size, alpha in prov_styles:
            sub = valid_df[mask]
            ax.scatter(
                sub["PP_TARGET_BAR"],
                sub[col_pred],
                label=label if idx == 0 else None,
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors="black" if marker in ("D", "*", "P") else "none",
                linewidths=0.6,
            )

        ax.plot([315, 375], [315, 375], "k--", lw=1.2, label="1:1 Parity Line" if idx == 0 else None)

        # Annotate Train RMSE, Val RMSE, and Test 20/21 MAE
        part_m = eval_tables["partition_metrics"]
        m_tr = part_m[(part_m["model_key"] == key) & (part_m["partition_subset"] == "Training (Valid)")].iloc[0]
        m_val = part_m[(part_m["model_key"] == key) & (part_m["partition_subset"] == "Validation (Valid)")].iloc[0]
        m_t2021 = part_m[(part_m["model_key"] == key) & (part_m["partition_subset"] == "Testing (Measured Anchors Only: Tests 20 & 21)")].iloc[0]

        stats_text = (
            f"Train: RMSE={m_tr['RMSE_bar']:.2f} bar, R2={m_tr['R2']:.3f}\n"
            f"Val (n=34): RMSE={m_val['RMSE_bar']:.2f} bar, MAE={m_val['MAE_bar']:.2f} bar\n"
            f"Tests 20 & 21: RMSE={m_t2021['RMSE_bar']:.2f} bar"
        )
        ax.text(
            0.04,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )

        ax.set_title(f"{info['display_name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual / Target Pore Pressure (bar)", fontsize=9.5)
        ax.set_ylabel("Predicted Pore Pressure (bar)", fontsize=9.5)
        ax.set_xlim(315, 375)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Use 6th subplot for legend and physical benchmark bar chart
    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
         frameon=True,
        fontsize=9.5,
        title="Partition and Target Provenance",
        title_fontsize=10.5,
    )

    fig.suptitle(
        "Well 15/9-F-12: Actual vs. Predicted Pore Pressure Across Candidate Models",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_path = figures_dir / "actual_vs_predicted.png"
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)
    generated_files.append(fig1_path)

    # -------------------------------------------------------------------------
    # Figure 2: Residuals vs. Depth and Residual Distributions
    # -------------------------------------------------------------------------
    fig, (ax_depth, ax_box) = plt.subplots(1, 2, figsize=(15, 8.5), dpi=200, gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Left panel: Residual (y_true - y_pred) vs. Depth (MD)
    for key in model_keys:
        info = trained_registry[key]
        col_res = f"RESIDUAL_{key.upper()}_BAR"
        # Plot Training + Validation + Testing separately so lines don't bridge the excluded gap
        for part_name in ["Training", "Validation", "Testing"]:
            sub = valid_df[valid_df["PARTITION"] == part_name]
            ax_depth.plot(
                sub[col_res],
                sub["DEPTH"],
                color=model_colors[key],
                lw=1.2 if part_name == "Training" else 1.8,
                alpha=0.85,
                label=info["short_name"] if part_name == "Training" else None,
            )

    ax_depth.axvline(0.0, color="black", linestyle="--", lw=1.1)
    ax_depth.axhspan(3326.0, 3386.5, color="#dddddd", alpha=0.6, label="Excluded Transition (3326.0-3386.5 m)")
    ax_depth.axhline(3316.5, color="#555555", linestyle=":", lw=1.0)
    ax_depth.axhline(3320.796, color="#0055aa", linestyle="-.", lw=1.0, label="Train / Val Split (3320.80 m)")
    ax_depth.axhline(3362.249, color="#aa5500", linestyle="-.", lw=1.0, label="Val / Test Split (3362.25 m)")

    ax_depth.invert_yaxis()
    ax_depth.set_xlabel("Residual: Target minus Predicted Pressure (bar)", fontsize=10.5)
    ax_depth.set_ylabel("Measured Depth (m RKB)", fontsize=10.5)
    ax_depth.set_title("(a) Pore-Pressure Residuals vs. Measured Depth", fontsize=11.5, fontweight="bold")
    ax_depth.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax_depth.grid(True, linestyle=":", alpha=0.6)

    # Right panel: Residual Boxplots across Training (n=1268) and Validation (n=34)
    positions_tr = np.arange(len(model_keys)) * 2.2
    positions_val = positions_tr + 0.75

    tr_res_data = [
        valid_df[valid_df["PARTITION"] == "Training"][f"RESIDUAL_{k.upper()}_BAR"].dropna().values
        for k in model_keys
    ]
    val_res_data = [
        valid_df[valid_df["PARTITION"] == "Validation"][f"RESIDUAL_{k.upper()}_BAR"].dropna().values
        for k in model_keys
    ]

    bp1 = ax_box.boxplot(
        tr_res_data,
        positions=positions_tr,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#4c72b0", alpha=0.75),
        medianprops=dict(color="black", lw=1.5),
    )
    bp2 = ax_box.boxplot(
        val_res_data,
        positions=positions_val,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#dd8452", alpha=0.80),
        medianprops=dict(color="black", lw=1.5),
    )

    ax_box.axhline(0.0, color="black", linestyle="--", lw=1.1)
    ax_box.set_xticks(positions_tr + 0.375)
    ax_box.set_xticklabels([trained_registry[k]["short_name"] for k in model_keys], rotation=15, fontsize=9.5)
    ax_box.set_ylabel("Residual: Target minus Predicted Pressure (bar)", fontsize=10.5)
    ax_box.set_title("(b) Residual Distribution (Training vs. Validation)", fontsize=11.5, fontweight="bold")
    ax_box.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Training (n=1268)", "Validation Lower Sleipner (n=34)"], loc="upper left", fontsize=9.5)
    ax_box.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Well 15/9-F-12: Residual Distribution and Depth-Extrapolation Diagnostics",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = figures_dir / "residuals_vs_depth_and_dist.png"
    fig.savefig(fig2_path, dpi=200)
    plt.close(fig)
    generated_files.append(fig2_path)

    # -------------------------------------------------------------------------
    # Figure 3: Continuous Borehole Pore-Pressure Profiles vs. Depth
    # -------------------------------------------------------------------------
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 9.5), dpi=200, sharey=True)

    # Panel (a): Full physical context including Hydrostatic, Overburden, Target, and Models
    ax_full.plot(pred_df["PH_HYDROSTATIC_BAR"], pred_df["DEPTH"], color="#0077bb", linestyle="--", lw=1.4, label="Hydrostatic Baseline (1.025 g/cm3)")
    ax_full.plot(pred_df["SV_OVERBURDEN_BAR"], pred_df["DEPTH"], color="#333333", linestyle="-.", lw=1.5, label="Composite Overburden Stress (Sv)")
    ax_full.plot(pred_df["PP_TARGET_BAR"], pred_df["DEPTH"], color="black", lw=2.4, label="Primary Target Profile (Valid Intervals)")

    for key in model_keys:
        info = trained_registry[key]
        ax_full.plot(
            pred_df[f"PRED_{key.upper()}_BAR"],
            pred_df["DEPTH"],
            color=model_colors[key],
            lw=1.1,
            alpha=0.80,
            label=f"Predicted: {info['short_name']}",
        )

    ax_full.axhspan(3326.0, 3386.5, color="#dddddd", alpha=0.55, label="Excluded Transition (3326.0-3386.5 m)")
    ax_full.invert_yaxis()
    ax_full.set_xlabel("Pressure / Stress (bar)", fontsize=10.5)
    ax_full.set_ylabel("Measured Depth (m RKB)", fontsize=10.5)
    ax_full.set_title("(a) Full Borehole Pressure Regime Context", fontsize=11.5, fontweight="bold")
    ax_full.legend(loc="lower left", fontsize=8.2, framealpha=0.92)
    ax_full.grid(True, linestyle=":", alpha=0.6)

    # Panel (b): Zoomed Pore-Pressure Window (315 to 385 bar) showing discrete stations clearly
    ax_zoom.plot(pred_df["PP_TARGET_BAR"], pred_df["DEPTH"], color="black", lw=2.5, zorder=4, label="Target Profile (Valid)")

    for key in model_keys:
        info = trained_registry[key]
        ax_zoom.plot(
            pred_df[f"PRED_{key.upper()}_BAR"],
            pred_df["DEPTH"],
            color=model_colors[key],
            lw=1.35,
            alpha=0.85,
            zorder=3,
            label=f"{info['short_name']}",
        )

    # Plot discrete Schlumberger stations
    st_train = station_df[station_df["test_no"].isin(list(range(1, 16)))]
    st_val16 = station_df[station_df["test_no"] == 16]
    st_test20_21 = station_df[station_df["test_no"].isin([20, 21])]
    st_test19 = station_df[station_df["test_no"] == 19]

    ax_zoom.scatter(
        st_train["reconciled_station_p_bar"],
        st_train["station_md_m"],
        c="#003366",
        marker="o",
        s=42,
        zorder=6,
        edgecolors="white",
        linewidths=0.6,
        label="Train Pretests (Tests 1-15, incl. T13 Plot 334.16 bar)",
    )
    ax_zoom.scatter(
        st_val16["reconciled_station_p_bar"],
        st_val16["station_md_m"],
        c="#ff6600",
        marker="*",
        s=150,
        zorder=7,
        edgecolors="black",
        linewidths=0.7,
        label="Val Checkpoint: Test 16 (341.71 bar)",
    )
    ax_zoom.scatter(
        st_test20_21["reconciled_station_p_bar"],
        st_test20_21["station_md_m"],
        c="#cc0000",
        marker="P",
        s=110,
        zorder=7,
        edgecolors="black",
        linewidths=0.7,
        label="Test Anchors: Tests 20 & 21 (368.71, 368.82 bar)",
    )
    ax_zoom.scatter(
        st_test19["reconciled_station_p_bar"],
        st_test19["station_md_m"],
        c="#aa00aa",
        marker="X",
        s=95,
        zorder=7,
        edgecolors="black",
        linewidths=0.7,
        label="Excluded Memory Obs: Test 19 (357.71 bar)",
    )

    ax_zoom.axhspan(3326.0, 3386.5, color="#dddddd", alpha=0.55)
    ax_zoom.axhline(3316.5, color="#555555", linestyle=":", lw=1.1, label="Top Sleipner (3316.5 m)")
    ax_zoom.axhline(3320.796, color="#0055aa", linestyle="-.", lw=1.1, label="Train / Val Boundary (3320.80 m)")
    ax_zoom.axhline(3362.249, color="#aa5500", linestyle="-.", lw=1.1, label="Val / Test Boundary (3362.25 m)")
    ax_zoom.axhline(3386.5, color="#880000", linestyle=":", lw=1.1, label="Top Evaluated Skagerrak (3386.5 m)")

    ax_zoom.set_xlim(312, 385)
    ax_zoom.set_xlabel("Pore Pressure (bar)", fontsize=10.5)
    ax_zoom.set_title("(b) Zoomed Pore-Pressure Profile and Pretest Stations", fontsize=11.5, fontweight="bold")
    ax_zoom.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    ax_zoom.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Well 15/9-F-12: Continuous Borehole Pore-Pressure Predictions vs. Measured Pretests",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_path = figures_dir / "borehole_pressure_profiles.png"
    fig.savefig(fig3_path, dpi=200)
    plt.close(fig)
    generated_files.append(fig3_path)

    # -------------------------------------------------------------------------
    # Figure 4: Native Tree Importance and Model-Independent Permutation Importance
    # -------------------------------------------------------------------------
    fig, (ax_nat, ax_perm) = plt.subplots(1, 2, figsize=(15.5, 7.5), dpi=200)

    predictor_cols = data_bundle["predictor_cols"]
    # Order features by mean permutation importance across models
    mean_perm_by_feat = (
        feat_df.groupby("feature")["perm_train_importance_pct"]
        .mean()
        .sort_values(ascending=True)
    )
    ordered_feats = mean_perm_by_feat.index.tolist()
    y_pos = np.arange(len(ordered_feats))

    # Left panel: Native importance (%) for RF, CatBoost, GradientBoosting (+ MLR standardized coef %)
    nat_keys = ["random_forest", "catboost", "gradient_boosting", "mlr"]
    bar_h = 0.20
    for i, k in enumerate(nat_keys):
        sub = feat_df[feat_df["model_key"] == k].set_index("feature").loc[ordered_feats]
        offset = (i - 1.5) * bar_h
        ax_nat.barh(
            y_pos + offset,
            sub["native_importance_pct"].values,
            height=bar_h,
            color=model_colors[k],
            alpha=0.85,
            label=trained_registry[k]["short_name"],
        )

    ax_nat.set_yticks(y_pos)
    ax_nat.set_yticklabels(ordered_feats, fontsize=9.5, fontweight="bold")
    ax_nat.set_xlabel("Normalized Native Feature Importance / |Std. Coef| (%)", fontsize=10)
    ax_nat.set_title("(a) Native Model Feature Importance (Trees & MLR)", fontsize=11.5, fontweight="bold")
    ax_nat.legend(loc="lower right", fontsize=9)
    ax_nat.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Right panel: Normalized Permutation Importance (% of total RMSE increase on Training set) across all 5 models
    bar_h5 = 0.16
    for i, k in enumerate(model_keys):
        sub = feat_df[feat_df["model_key"] == k].set_index("feature").loc[ordered_feats]
        offset = (i - 2.0) * bar_h5
        ax_perm.barh(
            y_pos + offset,
            sub["perm_train_importance_pct"].values,
            height=bar_h5,
            color=model_colors[k],
            alpha=0.85,
            label=trained_registry[k]["short_name"],
        )

    ax_perm.set_yticks(y_pos)
    ax_perm.set_yticklabels(ordered_feats, fontsize=9.5, fontweight="bold")
    ax_perm.set_xlabel("Normalized Permutation Importance (% of RMSE Increase)", fontsize=10)
    ax_perm.set_title("(b) Model-Independent Permutation Importance (All 5 Models)", fontsize=11.5, fontweight="bold")
    ax_perm.legend(loc="lower right", fontsize=9)
    ax_perm.grid(True, axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(
        "Well 15/9-F-12: Petrophysical and Drilling Feature Importance Comparison",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig4_path = figures_dir / "feature_importance_comparison.png"
    fig.savefig(fig4_path, dpi=200)
    plt.close(fig)
    generated_files.append(fig4_path)

    # -------------------------------------------------------------------------
    # Figure 5: Learning Curves (Iterative Convergence for Boosting & MLP)
    # -------------------------------------------------------------------------
    fig, (ax_boost, ax_mlp) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=200)

    X_tr_raw = data_bundle["X_train_raw"]
    y_tr = data_bundle["y_train"]
    X_val_raw_v = data_bundle["X_val_raw"].iloc[data_bundle["val_valid_mask"]]
    y_val_v = data_bundle["y_val"][data_bundle["val_valid_mask"]]

    # Panel (a): Boosting staged RMSE curves (CatBoost & GradientBoosting)
    # GradientBoosting staged_predict
    gb_model = trained_registry["gradient_boosting"]["model"].model
    gb_tr_rmse = [
        float(np.sqrt(mean_squared_error(y_tr, yp)))
        for yp in gb_model.staged_predict(X_tr_raw)
    ]
    gb_val_rmse = [
        float(np.sqrt(mean_squared_error(y_val_v, yp)))
        for yp in gb_model.staged_predict(X_val_raw_v)
    ]
    iters_gb = np.arange(1, len(gb_tr_rmse) + 1)

    # CatBoost staged predictions
    cb_model = trained_registry["catboost"]["model"].model
    cb_iters = np.arange(10, cb_model.tree_count_ + 1, 10)
    cb_tr_rmse = [
        float(np.sqrt(mean_squared_error(y_tr, cb_model.predict(X_tr_raw, ntree_end=int(nt)))))
        for nt in cb_iters
    ]
    cb_val_rmse = [
        float(np.sqrt(mean_squared_error(y_val_v, cb_model.predict(X_val_raw_v, ntree_end=int(nt)))))
        for nt in cb_iters
    ]

    ax_boost.plot(iters_gb, gb_tr_rmse, color=model_colors["gradient_boosting"], lw=1.8, label="Gradient Boosting - Train RMSE")
    ax_boost.plot(iters_gb, gb_val_rmse, color=model_colors["gradient_boosting"], linestyle="--", lw=1.8, label="Gradient Boosting - Val (Lower Sleipner) RMSE")
    ax_boost.plot(cb_iters, cb_tr_rmse, color=model_colors["catboost"], lw=1.8, label="CatBoost - Train RMSE")
    ax_boost.plot(cb_iters, cb_val_rmse, color=model_colors["catboost"], linestyle="--", lw=1.8, label="CatBoost - Val (Lower Sleipner) RMSE")

    ax_boost.set_xlabel("Number of Boosting Trees / Iterations", fontsize=10.5)
    ax_boost.set_ylabel("Root Mean Squared Error (bar)", fontsize=10.5)
    ax_boost.set_title("(a) Boosting Convergence (Train vs. Out-of-Sample Val)", fontsize=11.5, fontweight="bold")
    ax_boost.legend(loc="center right", fontsize=9)
    ax_boost.grid(True, linestyle=":", alpha=0.6)

    # Panel (b): MLP Training Loss Curve and Internal Holdout R2 Score
    mlp_sklearn = trained_registry["mlp"]["model"].model
    epochs_mlp = np.arange(1, len(mlp_sklearn.loss_curve_) + 1)
    ax_mlp.plot(
        epochs_mlp,
        np.sqrt(2.0 * np.asarray(mlp_sklearn.loss_curve_)),
        color=model_colors["mlp"],
        lw=2.0,
        label="MLP Training Approx. RMSE (sqrt(2 * loss))",
    )
    if hasattr(mlp_sklearn, "best_loss_") and mlp_sklearn.best_loss_ is not None:
        ax_mlp.axhline(
            float(np.sqrt(2.0 * mlp_sklearn.best_loss_)),
            color="#333333",
            linestyle=":",
            lw=1.2,
            label=f"Best Train RMSE ({np.sqrt(2.0 * mlp_sklearn.best_loss_):.2f} bar)",
        )

    ax_mlp.set_xlabel("Training Epochs (Adam Solver)", fontsize=10.5)
    ax_mlp.set_ylabel("Training RMSE Proxy (bar)", fontsize=10.5)
    ax_mlp.set_title(
        f"(b) MLP Training Convergence (Stopped at Epoch {mlp_sklearn.n_iter_})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_mlp.legend(loc="upper right", fontsize=9.5)
    ax_mlp.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Well 15/9-F-12: Model Training Convergence and Overfitting Diagnostics",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig5_path = figures_dir / "learning_curves.png"
    fig.savefig(fig5_path, dpi=200)
    plt.close(fig)
    generated_files.append(fig5_path)

    return generated_files


def run_training_pipeline(base_dir: Path | None = None) -> Dict[str, Any]:
    """Execute the complete model training, evaluation, and plotting workflow."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    models_config_path = base_dir / "configs" / "models.yaml"
    models_out_dir = base_dir / "outputs" / "models"
    metrics_dir = base_dir / "outputs" / "metrics"
    figures_dir = base_dir / "outputs" / "figures"
    tables_dir = base_dir / "tables"
    audit_dir = base_dir / "outputs" / "audit"

    data_bundle = prepare_training_data(base_dir)
    trained_registry = train_all_models(data_bundle, models_config_path, models_out_dir)
    eval_tables = evaluate_all_models(
        data_bundle, trained_registry, metrics_dir, tables_dir, audit_dir
    )
    figure_paths = generate_diagnostic_figures(
        data_bundle, trained_registry, eval_tables, figures_dir
    )

    return {
        "data_bundle": data_bundle,
        "trained_registry": trained_registry,
        "eval_tables": eval_tables,
        "figure_paths": figure_paths,
    }
