"""Pipeline for executing controlled experiments A, B, and C.

Strictly preserves:
1. 70:15:15 contiguous depth split (Train: 1268, Val: 272, Test: 271).
2. All scalers and transformations fitted exclusively on training rows.
3. Zero contamination of validation or testing data in preprocessing or model fitting.
4. Exclusion of unsupported transition rows (MD 3326.0 to 3386.5 m) from metrics.
5. Tests 20 & 21 held out strictly as final physical evaluation stations.
6. Original baseline models and outputs remain completely unmodified.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

from src.data.validation import DEPTH_COL
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.residuals import analyze_residuals
from src.experiments.features import (
    build_experiment_a_features,
    build_experiment_b_features,
    build_experiment_c_features,
)
from src.models import (
    MultipleLinearRegressionModel,
    RandomForestModel,
    CatBoostModel,
    GradientBoostingModel,
    MLPModel,
)

matplotlib.use("Agg")
RANDOM_SEED = 42
LOG_TRANSFORM_COLS = ["K", "RS", "RD", "RT"]


def _save_dual(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    records = json.loads(df.to_json(orient="records"))
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)


def run_single_experiment(
    exp_key: str,
    exp_name: str,
    X_all_raw: pd.DataFrame,
    df_targ: pd.DataFrame,
    station_comp_df: pd.DataFrame,
    models_cfg: Dict[str, Any],
    out_models_dir: Path,
) -> Dict[str, Any]:
    """Train the 5 candidate models on a given experimental feature set and compute full metrics."""
    train_mask = (df_targ["PARTITION"] == "Training").values
    val_mask = (df_targ["PARTITION"] == "Validation").values
    test_mask = (df_targ["PARTITION"] == "Testing").values
    valid_mask = df_targ["IS_TARGET_VALID"].astype(bool).values

    y_all = df_targ["PP_TARGET_BAR"].values.astype(float)
    y_tr_valid = y_all[train_mask]

    # Preprocessing: Scaler fitted exclusively on Training partition
    X_all_log = X_all_raw.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in X_all_log.columns:
            X_all_log[col] = np.log10(np.clip(X_all_log[col].values, a_min=1e-6, a_max=None))

    scaler = StandardScaler()
    scaler.fit(X_all_log.iloc[train_mask])

    X_all_scaled = pd.DataFrame(
        scaler.transform(X_all_log),
        columns=X_all_raw.columns,
        index=X_all_raw.index,
    )

    X_tr_raw = X_all_raw.iloc[train_mask].copy().reset_index(drop=True)
    X_tr_scaled = X_all_scaled.iloc[train_mask].copy().reset_index(drop=True)

    model_specs = [
        {"key": "mlr", "name": "MLR", "cls": MultipleLinearRegressionModel, "use_scaled": True, "params": models_cfg.get("mlr", {})},
        {"key": "random_forest", "name": "Random Forest", "cls": RandomForestModel, "use_scaled": False, "params": models_cfg.get("random_forest", {})},
        {"key": "catboost", "name": "CatBoost", "cls": CatBoostModel, "use_scaled": False, "params": models_cfg.get("catboost", {})},
        {"key": "gradient_boosting", "name": "Gradient Boosting", "cls": GradientBoostingModel, "use_scaled": False, "params": models_cfg.get("gradient_boosting", {})},
        {"key": "mlp", "name": "MLP", "cls": MLPModel, "use_scaled": True, "params": models_cfg.get("mlp", {})},
    ]

    trained_models = {}
    predictions_dict = {}

    out_models_dir.mkdir(parents=True, exist_ok=True)

    for spec in model_specs:
        m_key = spec["key"]
        m_instance = spec["cls"](params=spec["params"])

        X_fit = X_tr_scaled if spec["use_scaled"] else X_tr_raw
        X_predict = X_all_scaled if spec["use_scaled"] else X_all_raw

        m_instance.fit(X_fit, y_tr_valid)
        y_pred = m_instance.predict(X_predict)

        model_path = out_models_dir / f"{exp_key}_{m_key}_model.joblib"
        m_instance.save(model_path)

        trained_models[m_key] = m_instance
        predictions_dict[m_key] = y_pred

    # Save fitted scaler
    scaler_path = out_models_dir / f"{exp_key}_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Master predictions DataFrame for this experiment
    exp_pred_df = df_targ[
        [
            "observation_index",
            "DEPTH",
            "TVD_MSL",
            "FORMATION",
            "PARTITION",
            "IS_TARGET_VALID",
            "TARGET_PROVENANCE",
            "PP_TARGET_BAR",
            "MEASURED_TEST_NO",
            "MEASURED_PRESSURE_BAR",
        ]
    ].copy()

    for m_key, y_pred in predictions_dict.items():
        exp_pred_df[f"PRED_{m_key.upper()}_BAR"] = np.round(y_pred, 4)
        exp_pred_df[f"RESIDUAL_{m_key.upper()}_BAR"] = np.where(valid_mask, np.round(y_all - y_pred, 4), np.nan)

    # Partition Metrics
    partition_subsets = [
        ("Training (Valid)", train_mask & valid_mask),
        ("Validation (Valid)", val_mask & valid_mask),
        ("Testing (Measured Anchors Only: Tests 20 & 21)", test_mask & (df_targ["TARGET_PROVENANCE"].values == "measured_anchor")),
        ("Testing (Provisional Skagerrak Assumption Only)", test_mask & (df_targ["TARGET_PROVENANCE"].values == "provisional_assumption")),
        ("Full Dataset (All Valid Targets)", valid_mask),
    ]

    part_metrics_rows = []
    for spec in model_specs:
        m_key = spec["key"]
        y_pred = predictions_dict[m_key]
        for sub_label, mask in partition_subsets:
            m = compute_regression_metrics(y_all[mask], y_pred[mask])
            res_stats = analyze_residuals(y_all[mask] - y_pred[mask])
            part_metrics_rows.append(
                {
                    "experiment_key": exp_key,
                    "experiment_name": exp_name,
                    "model_key": m_key,
                    "model_name": spec["name"],
                    "partition_subset": sub_label,
                    "n_valid": int(mask.sum()),
                    "MAE_bar": round(m["MAE"], 4),
                    "RMSE_bar": round(m["RMSE"], 4),
                    "R2": round(m["R2"], 4) if np.isfinite(m["R2"]) else np.nan,
                    "mean_residual_bar": round(res_stats.get("mean_residual", np.nan), 4),
                    "std_residual_bar": round(res_stats.get("std_residual", np.nan), 4),
                    "MaxAE_bar": round(m["MaxAE"], 4),
                }
            )

    df_part_metrics = pd.DataFrame(part_metrics_rows)

    # Provenance Metrics
    prov_col = df_targ["TARGET_PROVENANCE"].values
    prov_subsets = [
        ("measured_anchor", "Training (Tests 1-12, 14, 15)", (prov_col == "measured_anchor") & train_mask),
        ("measured_anchor", "Validation (Test 16)", (prov_col == "measured_anchor") & val_mask),
        ("measured_anchor", "Testing (Tests 20 & 21)", (prov_col == "measured_anchor") & test_mask),
        ("physics_derived", "Training (Hugin/Sleipner)", (prov_col == "physics_derived") & train_mask),
        ("physics_derived", "Validation (Lower Sleipner)", (prov_col == "physics_derived") & val_mask),
        ("provisional_assumption", "Testing (Skagerrak)", prov_col == "provisional_assumption"),
    ]

    prov_metrics_rows = []
    for spec in model_specs:
        m_key = spec["key"]
        y_pred = predictions_dict[m_key]
        for prov_label, part_label, mask in prov_subsets:
            m = compute_regression_metrics(y_all[mask], y_pred[mask])
            res = y_all[mask] - y_pred[mask]
            prov_metrics_rows.append(
                {
                    "experiment_key": exp_key,
                    "experiment_name": exp_name,
                    "model_key": m_key,
                    "model_name": spec["name"],
                    "target_provenance": prov_label,
                    "partition_interval": part_label,
                    "n_observations": int(mask.sum()),
                    "MAE_bar": round(m["MAE"], 4),
                    "RMSE_bar": round(m["RMSE"], 4),
                    "mean_residual_bar": round(float(np.mean(res)), 4),
                    "MaxAE_bar": round(m["MaxAE"], 4),
                }
            )

    df_prov_metrics = pd.DataFrame(prov_metrics_rows)

    # Station Evaluation
    station_eval_rows = []
    for _, st in station_comp_df.iterrows():
        test_no = int(st["test_no"])
        grid_md = float(st["grid_md"])
        idx = int(np.argmin(np.abs(df_targ["DEPTH"].values - grid_md)))
        reconciled_p = st["reconciled_station_p_bar"]
        has_valid_p = pd.notna(reconciled_p)

        row_dict: Dict[str, Any] = {
            "experiment_key": exp_key,
            "test_no": test_no,
            "station_md_m": float(st["station_md"]),
            "grid_md_m": float(df_targ.loc[idx, "DEPTH"]),
            "partition": str(st["partition"]),
            "station_role": str(st["station_role"]),
            "reconciled_station_p_bar": float(reconciled_p) if has_valid_p else np.nan,
        }

        for spec in model_specs:
            m_key = spec["key"]
            pred_val = float(predictions_dict[m_key][idx])
            row_dict[f"pred_{m_key}_bar"] = round(pred_val, 4)
            if has_valid_p:
                err = float(reconciled_p) - pred_val
                row_dict[f"residual_{m_key}_bar"] = round(err, 4)
            else:
                row_dict[f"residual_{m_key}_bar"] = np.nan

        station_eval_rows.append(row_dict)

    df_station_eval = pd.DataFrame(station_eval_rows)

    # Station Summary Benchmarks
    t16_row = df_station_eval[df_station_eval["test_no"] == 16].iloc[0]
    t20_row = df_station_eval[df_station_eval["test_no"] == 20].iloc[0]
    t21_row = df_station_eval[df_station_eval["test_no"] == 21].iloc[0]
    t19_row = df_station_eval[df_station_eval["test_no"] == 19].iloc[0]
    tr_stations = df_station_eval[df_station_eval["test_no"].isin(list(range(1, 16)))]

    st_benchmark_rows = []
    for spec in model_specs:
        m_key = spec["key"]
        tr_true = tr_stations["reconciled_station_p_bar"].values
        tr_pred = tr_stations[f"pred_{m_key}_bar"].values
        tr_rmse = float(np.sqrt(np.mean((tr_true - tr_pred) ** 2)))

        t20_21_true = np.array([t20_row["reconciled_station_p_bar"], t21_row["reconciled_station_p_bar"]])
        t20_21_pred = np.array([t20_row[f"pred_{m_key}_bar"], t21_row[f"pred_{m_key}_bar"]])
        t20_21_mae = float(np.mean(np.abs(t20_21_true - t20_21_pred)))
        t20_21_rmse = float(np.sqrt(np.mean((t20_21_true - t20_21_pred) ** 2)))

        st_benchmark_rows.append(
            {
                "experiment_key": exp_key,
                "experiment_name": exp_name,
                "model_key": m_key,
                "model_name": spec["name"],
                "train_15_stations_RMSE_bar": round(tr_rmse, 4),
                "val_test16_measured_bar": round(float(t16_row["reconciled_station_p_bar"]), 3),
                "val_test16_pred_bar": round(float(t16_row[f"pred_{m_key}_bar"]), 4),
                "val_test16_residual_bar": round(float(t16_row[f"residual_{m_key}_bar"]), 4),
                "test20_pred_bar": round(float(t20_row[f"pred_{m_key}_bar"]), 4),
                "test20_residual_bar": round(float(t20_row[f"residual_{m_key}_bar"]), 4),
                "test21_pred_bar": round(float(t21_row[f"pred_{m_key}_bar"]), 4),
                "test21_residual_bar": round(float(t21_row[f"residual_{m_key}_bar"]), 4),
                "test20_21_MAE_bar": round(t20_21_mae, 4),
                "test20_21_RMSE_bar": round(t20_21_rmse, 4),
                "excluded_test19_pred_bar": round(float(t19_row[f"pred_{m_key}_bar"]), 4),
                "excluded_test19_diff_bar": round(float(t19_row[f"residual_{m_key}_bar"]), 4),
            }
        )

    df_st_benchmarks = pd.DataFrame(st_benchmark_rows)

    return {
        "predictions_df": exp_pred_df,
        "partition_metrics": df_part_metrics,
        "provenance_metrics": df_prov_metrics,
        "station_evaluation": df_station_eval,
        "station_benchmarks": df_st_benchmarks,
        "trained_models": trained_models,
        "features": list(X_all_raw.columns),
    }


def generate_experiment_figures(
    df_targ: pd.DataFrame,
    baseline_pred_df: pd.DataFrame,
    exp_a_pred_df: pd.DataFrame,
    exp_b_pred_df: pd.DataFrame,
    exp_c_pred_df: pd.DataFrame,
    station_comp_df: pd.DataFrame,
    figures_dir: Path,
    exp_b_no_dt_pred_df: pd.DataFrame | None = None,
) -> List[Path]:
    """Generate diagnostic comparative figures across baseline and experiments A, B, B (no DT ratio), and C."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: List[Path] = []

    # 1. Multi-experiment borehole comparison for CatBoost and MLP
    fig, (ax_cb, ax_mlp) = plt.subplots(1, 2, figsize=(16, 9.5), dpi=200, sharey=True)

    depths = df_targ["DEPTH"].values
    targ_p = df_targ["PP_TARGET_BAR"].values

    st_train = station_comp_df[station_comp_df["test_no"].isin(list(range(1, 16)))]
    st_val16 = station_comp_df[station_comp_df["test_no"] == 16]
    st_test = station_comp_df[station_comp_df["test_no"].isin([20, 21])]

    for ax, m_key, m_title in [(ax_cb, "CATBOOST", "CatBoost Regressor"), (ax_mlp, "MLP", "Multi-Layer Perceptron")]:
        ax.plot(targ_p, depths, color="black", lw=2.4, label="Target Profile (Valid)")
        ax.plot(baseline_pred_df[f"PRED_{m_key}_BAR"], depths, color="#7f7f7f", linestyle="--", lw=1.2, label="Baseline (Logs-Only)")
        ax.plot(exp_a_pred_df[f"PRED_{m_key}_BAR"], depths, color="#1f77b4", lw=1.3, label="Exp A (+TVD_MSL)")
        ax.plot(exp_b_pred_df[f"PRED_{m_key}_BAR"], depths, color="#2ca02c", lw=1.3, label="Exp B (+Physics: Sv, Ph, Eff, DT_ratio)")
        if exp_b_no_dt_pred_df is not None:
            ax.plot(exp_b_no_dt_pred_df[f"PRED_{m_key}_BAR"], depths, color="#17becf", linestyle="-.", lw=1.3, label="Exp B Sensitivity (No DT_ratio)")
        ax.plot(exp_c_pred_df[f"PRED_{m_key}_BAR"], depths, color="#d62728", lw=1.3, label="Exp C (+Formation Flags)")

        # Pretest stations
        ax.scatter(st_train["reconciled_station_p_bar"], st_train["station_md"], c="#003366", s=35, zorder=6, label="Train Pretests (1-15)")
        ax.scatter(st_val16["reconciled_station_p_bar"], st_val16["station_md"], c="#ff6600", marker="*", s=140, zorder=7, label="Val Checkpoint (Test 16)")
        ax.scatter(st_test["reconciled_station_p_bar"], st_test["station_md"], c="#cc0000", marker="P", s=95, zorder=7, label="Test Anchors (20 & 21)")

        ax.axhspan(3326.0, 3386.5, color="#e0e0e0", alpha=0.55, label="Excluded Transition (3326-3386.5 m)")
        ax.axhline(3320.796, color="#0055aa", linestyle="-.", lw=1.0)
        ax.axhline(3362.249, color="#aa5500", linestyle="-.", lw=1.0)

        ax.set_xlim(310, 385)
        ax.set_xlabel("Pore Pressure (bar)", fontsize=10.5)
        ax.set_title(f"{m_title}: Controlled Experiments Comparison", fontsize=11.5, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax_cb.invert_yaxis()
    ax_cb.set_ylabel("Measured Depth (m RKB)", fontsize=10.5)

    fig.suptitle(
        "Well 15/9-F-12: Impact of Depth, Physics, and Formation Features Across Borehole",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpath1 = figures_dir / "all_experiments_borehole_comparison.png"
    fig.savefig(fpath1, dpi=200)
    plt.close(fig)
    generated_figures.append(fpath1)

    return generated_figures


def build_individual_station_errors(
    configs: List[Tuple[str, str, pd.DataFrame]],
    station_comp_df: pd.DataFrame,
    df_targ: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build individual physical station error tables for Tests 16, 20, 21, and 19.
    Returns:
    1. df_long: One row per (Experiment, Model, Station) with actual, predicted, residual, and absolute error.
    2. df_wide: Summary matrix with one row per (Experiment, Model) reporting metrics across stations.
    """
    models = [
        ("mlr", "MLR"),
        ("random_forest", "Random Forest"),
        ("catboost", "CatBoost"),
        ("gradient_boosting", "Gradient Boosting"),
        ("mlp", "MLP"),
    ]
    eval_tests = [16, 20, 21, 19]
    station_meta = {
        16: {"label": "Test 16", "formation": "Sleipner", "partition": "Validation", "role": "Unseen Validation Checkpoint"},
        20: {"label": "Test 20", "formation": "Skagerrak", "partition": "Testing", "role": "Testing Measured Anchor"},
        21: {"label": "Test 21", "formation": "Skagerrak", "partition": "Testing", "role": "Testing Measured Anchor"},
        19: {"label": "Test 19 (Excluded Memory)", "formation": "Skagerrak", "partition": "Testing", "role": "Excluded Conflicting Memory Pretest"},
    }

    long_rows = []
    wide_rows = []

    for exp_key, exp_name, pred_df in configs:
        for m_key, m_name in models:
            station_preds = {}
            for t in eval_tests:
                st_row = station_comp_df[station_comp_df["test_no"] == t].iloc[0]
                grid_md = float(st_row["grid_md"])
                idx = int(np.argmin(np.abs(df_targ["DEPTH"].values - grid_md)))
                p_actual = float(st_row["reconciled_station_p_bar"])
                p_pred = float(pred_df.loc[idx, f"PRED_{m_key.upper()}_BAR"])
                res = p_actual - p_pred
                abs_err = abs(res)

                station_preds[t] = {
                    "actual": p_actual,
                    "pred": p_pred,
                    "res": res,
                    "abs_err": abs_err,
                }

                long_rows.append({
                    "experiment_key": exp_key,
                    "experiment_name": exp_name,
                    "model_key": m_key,
                    "model_name": m_name,
                    "test_no": t,
                    "station_label": station_meta[t]["label"],
                    "formation": station_meta[t]["formation"],
                    "partition": station_meta[t]["partition"],
                    "station_role": station_meta[t]["role"],
                    "station_md_m": float(st_row["station_md"]),
                    "station_tvd_msl_m": float(df_targ.loc[idx, "TVD_MSL"]),
                    "actual_pressure_bar": round(p_actual, 3),
                    "predicted_pressure_bar": round(p_pred, 4),
                    "residual_bar": round(res, 4),
                    "abs_error_bar": round(abs_err, 4),
                })

            t20_21_mae = float(np.mean([station_preds[20]["abs_err"], station_preds[21]["abs_err"]]))
            t20_21_rmse = float(np.sqrt(np.mean([station_preds[20]["res"] ** 2, station_preds[21]["res"] ** 2])))

            wide_rows.append({
                "experiment_key": exp_key,
                "experiment_name": exp_name,
                "model_key": m_key,
                "model_name": m_name,
                "val_test16_actual_bar": round(station_preds[16]["actual"], 3),
                "val_test16_pred_bar": round(station_preds[16]["pred"], 4),
                "val_test16_residual_bar": round(station_preds[16]["res"], 4),
                "val_test16_abs_err_bar": round(station_preds[16]["abs_err"], 4),
                "test20_actual_bar": round(station_preds[20]["actual"], 3),
                "test20_pred_bar": round(station_preds[20]["pred"], 4),
                "test20_residual_bar": round(station_preds[20]["res"], 4),
                "test20_abs_err_bar": round(station_preds[20]["abs_err"], 4),
                "test21_actual_bar": round(station_preds[21]["actual"], 3),
                "test21_pred_bar": round(station_preds[21]["pred"], 4),
                "test21_residual_bar": round(station_preds[21]["res"], 4),
                "test21_abs_err_bar": round(station_preds[21]["abs_err"], 4),
                "test20_21_mae_bar": round(t20_21_mae, 4),
                "test20_21_rmse_bar": round(t20_21_rmse, 4),
                "excluded_test19_reference_bar": round(station_preds[19]["actual"], 3),
                "excluded_test19_pred_bar": round(station_preds[19]["pred"], 4),
                "excluded_test19_diff_bar": round(station_preds[19]["res"], 4),
                "excluded_test19_abs_diff_bar": round(station_preds[19]["abs_err"], 4),
            })

    df_long = pd.DataFrame(long_rows)
    df_wide = pd.DataFrame(wide_rows)
    return df_long, df_wide


def compute_mlp_depth_sensitivity(
    mlp_model: MLPModel,
    scaler: StandardScaler,
    feature_names: List[str],
    df_pred: pd.DataFrame,
    df_targ: pd.DataFrame,
    tvd_start_m: float = 2800.0,
    tvd_end_m: float = 3050.0,
    n_points: int = 251,
) -> pd.DataFrame:
    """
    Evaluate depth sensitivity profile of the Experiment A Depth-Aware MLP model.
    Sweeps TVD_MSL across the well interval while holding petrophysical logs
    at their median training values.
    """
    train_mask = (df_targ["PARTITION"] == "Training").values
    from src.data.validation import EXPECTED_PREDICTORS

    base_features = {}
    for col in EXPECTED_PREDICTORS:
        val = float(df_pred.loc[train_mask, col].median())
        base_features[col] = val

    tvd_sweep = np.linspace(tvd_start_m, tvd_end_m, n_points)
    sweep_df = pd.DataFrame([base_features] * n_points)
    sweep_df["TVD_MSL"] = tvd_sweep

    # Apply log10 and scaler transformation
    X_log = sweep_df.copy()
    for col in LOG_TRANSFORM_COLS:
        if col in X_log.columns:
            X_log[col] = np.log10(np.clip(X_log[col].values, a_min=1e-6, a_max=None))

    X_scaled = pd.DataFrame(scaler.transform(X_log), columns=feature_names)
    p_pred = mlp_model.predict(X_scaled)
    dP_dTVD = np.gradient(p_pred, tvd_sweep)

    regime = []
    for z in tvd_sweep:
        if z < 2819.38:
            regime.append("Pre-Training Shallow")
        elif z <= 2933.64:
            regime.append("Training Interval (Hugin & Upper Sleipner)")
        elif z <= 2958.51:
            regime.append("Validation Interval (Lower Sleipner)")
        elif z <= 2983.34:
            regime.append("Testing Interval (Skagerrak Compartment)")
        else:
            regime.append("Extrapolated Deep Sub-Skagerrak")

    return pd.DataFrame({
        "TVD_MSL_m": np.round(tvd_sweep, 2),
        "predicted_pressure_bar": np.round(p_pred, 4),
        "dP_dTVD_bar_per_m": np.round(dP_dTVD, 4),
        "equivalent_fluid_density_g_cm3": np.round(dP_dTVD / 0.0980665, 3),
        "stratigraphic_regime": regime,
    })


def run_all_experiments(base_dir: Path | None = None) -> Dict[str, Any]:
    """Execute complete controlled experiments suite."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    models_cfg_path = base_dir / "configs" / "models.yaml"
    with open(models_cfg_path, "r") as f:
        models_cfg = yaml.safe_load(f)["models"]

    df_pred = pd.read_csv(base_dir / "dataset" / "F12_predictor_dataset_1811.csv")
    df_targ = pd.read_csv(base_dir / "dataset" / "F12_targets_1811.csv")
    station_comp_df = pd.read_csv(base_dir / "outputs" / "audit" / "target_vs_measured_comparison.csv")
    ovb_sens_df = pd.read_csv(base_dir / "outputs" / "audit" / "overburden_sensitivity.csv")
    baseline_pred_df = pd.read_csv(base_dir / "outputs" / "metrics" / "model_predictions_1811.csv")

    exp_models_dir = base_dir / "outputs" / "experiments" / "models"
    exp_metrics_dir = base_dir / "outputs" / "experiments" / "metrics"
    exp_tables_dir = base_dir / "tables" / "experiments"
    exp_figures_dir = base_dir / "outputs" / "experiments" / "figures"

    exp_models_dir.mkdir(parents=True, exist_ok=True)
    exp_metrics_dir.mkdir(parents=True, exist_ok=True)
    exp_tables_dir.mkdir(parents=True, exist_ok=True)
    exp_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature sets
    X_exp_a = build_experiment_a_features(df_pred, df_targ)
    X_exp_b, meta_b = build_experiment_b_features(df_pred, df_targ, ovb_sens_df, include_dt_ratio=True)
    X_exp_b_no_dt, meta_b_no_dt = build_experiment_b_features(df_pred, df_targ, ovb_sens_df, include_dt_ratio=False)
    X_exp_c, meta_c = build_experiment_c_features(df_pred, df_targ)

    # Save feature metadata
    with open(exp_metrics_dir / "experiment_b_physics_features_metadata.json", "w") as f:
        json.dump(meta_b, f, indent=2)
    with open(exp_metrics_dir / "experiment_b_no_dtratio_physics_features_metadata.json", "w") as f:
        json.dump(meta_b_no_dt, f, indent=2)
    with open(exp_metrics_dir / "experiment_c_formation_features_metadata.json", "w") as f:
        json.dump(meta_c, f, indent=2)

    # 2. Run Experiment A (Depth-Aware)
    res_a = run_single_experiment(
        "exp_a", "Experiment A (Depth-Aware: +TVD_MSL)", X_exp_a, df_targ, station_comp_df, models_cfg, exp_models_dir
    )

    # 3. Run Experiment B (Physics-Informed with DT_ratio)
    res_b = run_single_experiment(
        "exp_b", "Experiment B (Physics-Informed: +Sv, +Ph, +sigma_v, +DT_nct_ratio)", X_exp_b, df_targ, station_comp_df, models_cfg, exp_models_dir
    )

    # 4. Run Experiment B Sensitivity (Physics-Informed without DT_ratio)
    res_b_no_dt = run_single_experiment(
        "exp_b_no_dtratio", "Experiment B (No DT_ratio: +Sv, +Ph, +sigma_v)", X_exp_b_no_dt, df_targ, station_comp_df, models_cfg, exp_models_dir
    )

    # 5. Run Experiment C (Formation-Aware)
    res_c = run_single_experiment(
        "exp_c", "Experiment C (Formation-Aware: +Stratigraphy, +Hardgrounds)", X_exp_c, df_targ, station_comp_df, models_cfg, exp_models_dir
    )

    # Save individual experiment predictions
    _save_dual(res_a["predictions_df"], exp_metrics_dir / "exp_a_predictions.csv", exp_metrics_dir / "exp_a_predictions.json")
    _save_dual(res_a["predictions_df"], exp_tables_dir / "exp_a_predictions.csv", exp_tables_dir / "exp_a_predictions.json")

    _save_dual(res_b["predictions_df"], exp_metrics_dir / "exp_b_predictions.csv", exp_metrics_dir / "exp_b_predictions.json")
    _save_dual(res_b["predictions_df"], exp_tables_dir / "exp_b_predictions.csv", exp_tables_dir / "exp_b_predictions.json")

    _save_dual(res_b_no_dt["predictions_df"], exp_metrics_dir / "exp_b_no_dtratio_predictions.csv", exp_metrics_dir / "exp_b_no_dtratio_predictions.json")
    _save_dual(res_b_no_dt["predictions_df"], exp_tables_dir / "exp_b_no_dtratio_predictions.csv", exp_tables_dir / "exp_b_no_dtratio_predictions.json")

    _save_dual(res_c["predictions_df"], exp_metrics_dir / "exp_c_predictions.csv", exp_metrics_dir / "exp_c_predictions.json")
    _save_dual(res_c["predictions_df"], exp_tables_dir / "exp_c_predictions.csv", exp_tables_dir / "exp_c_predictions.json")

    # Combine Partition Metrics across Baseline and Experiments
    baseline_part_metrics = pd.read_csv(base_dir / "outputs" / "metrics" / "partition_metrics.csv")
    baseline_part_metrics["experiment_key"] = "baseline"
    baseline_part_metrics["experiment_name"] = "Baseline (Logs-Only: 13 Features)"

    combined_part_metrics = pd.concat(
        [
            baseline_part_metrics[
                [
                    "experiment_key",
                    "experiment_name",
                    "model_key",
                    "model_name",
                    "partition_subset",
                    "n_valid",
                    "MAE_bar",
                    "RMSE_bar",
                    "R2",
                    "mean_residual_bar",
                    "MaxAE_bar",
                ]
            ],
            res_a["partition_metrics"][
                [
                    "experiment_key",
                    "experiment_name",
                    "model_key",
                    "model_name",
                    "partition_subset",
                    "n_valid",
                    "MAE_bar",
                    "RMSE_bar",
                    "R2",
                    "mean_residual_bar",
                    "MaxAE_bar",
                ]
            ],
            res_b["partition_metrics"][
                [
                    "experiment_key",
                    "experiment_name",
                    "model_key",
                    "model_name",
                    "partition_subset",
                    "n_valid",
                    "MAE_bar",
                    "RMSE_bar",
                    "R2",
                    "mean_residual_bar",
                    "MaxAE_bar",
                ]
            ],
            res_b_no_dt["partition_metrics"][
                [
                    "experiment_key",
                    "experiment_name",
                    "model_key",
                    "model_name",
                    "partition_subset",
                    "n_valid",
                    "MAE_bar",
                    "RMSE_bar",
                    "R2",
                    "mean_residual_bar",
                    "MaxAE_bar",
                ]
            ],
            res_c["partition_metrics"][
                [
                    "experiment_key",
                    "experiment_name",
                    "model_key",
                    "model_name",
                    "partition_subset",
                    "n_valid",
                    "MAE_bar",
                    "RMSE_bar",
                    "R2",
                    "mean_residual_bar",
                    "MaxAE_bar",
                ]
            ],
        ],
        ignore_index=True,
    )

    _save_dual(combined_part_metrics, exp_metrics_dir / "experiment_comparison_summary.csv", exp_metrics_dir / "experiment_comparison_summary.json")
    _save_dual(combined_part_metrics, exp_tables_dir / "experiment_comparison_summary.csv", exp_tables_dir / "experiment_comparison_summary.json")

    # Combine Station Benchmarks
    baseline_st_benchmarks = pd.read_csv(base_dir / "outputs" / "metrics" / "physical_station_benchmarks.csv")
    baseline_st_benchmarks["experiment_key"] = "baseline"
    baseline_st_benchmarks["experiment_name"] = "Baseline (Logs-Only)"

    st_bench_cols = [
        "experiment_key",
        "experiment_name",
        "model_key",
        "model_name",
        "train_15_stations_RMSE_bar",
        "val_test16_pred_bar",
        "val_test16_residual_bar",
        "test20_pred_bar",
        "test20_residual_bar",
        "test21_pred_bar",
        "test21_residual_bar",
        "test20_21_MAE_bar",
        "test20_21_RMSE_bar",
        "excluded_test19_pred_bar",
        "excluded_test19_diff_bar",
    ]

    combined_st_benchmarks = pd.concat(
        [
            baseline_st_benchmarks[st_bench_cols],
            res_a["station_benchmarks"][st_bench_cols],
            res_b["station_benchmarks"][st_bench_cols],
            res_b_no_dt["station_benchmarks"][st_bench_cols],
            res_c["station_benchmarks"][st_bench_cols],
        ],
        ignore_index=True,
    )

    _save_dual(combined_st_benchmarks, exp_metrics_dir / "experiment_station_benchmarks.csv", exp_metrics_dir / "experiment_station_benchmarks.json")
    _save_dual(combined_st_benchmarks, exp_tables_dir / "experiment_station_benchmarks.csv", exp_tables_dir / "experiment_station_benchmarks.json")

    # Combine Provenance Metrics
    baseline_prov_metrics = pd.read_csv(base_dir / "outputs" / "metrics" / "provenance_metrics.csv")
    baseline_prov_metrics["experiment_key"] = "baseline"
    baseline_prov_metrics["experiment_name"] = "Baseline (Logs-Only)"

    combined_prov_metrics = pd.concat(
        [
            baseline_prov_metrics,
            res_a["provenance_metrics"],
            res_b["provenance_metrics"],
            res_b_no_dt["provenance_metrics"],
            res_c["provenance_metrics"],
        ],
        ignore_index=True,
    )
    _save_dual(combined_prov_metrics, exp_metrics_dir / "experiment_provenance_metrics.csv", exp_metrics_dir / "experiment_provenance_metrics.json")
    _save_dual(combined_prov_metrics, exp_tables_dir / "experiment_provenance_metrics.csv", exp_tables_dir / "experiment_provenance_metrics.json")

    # Build Individual Physical Station Errors (Long and Wide Matrices)
    eval_configs = [
        ("baseline", "Baseline (Logs-Only: 13 Features)", baseline_pred_df),
        ("exp_a", "Experiment A (Depth-Aware: +TVD_MSL)", res_a["predictions_df"]),
        ("exp_b", "Experiment B (Physics-Informed: +Sv, +Ph, +sigma_v, +DT_nct_ratio)", res_b["predictions_df"]),
        ("exp_b_no_dtratio", "Experiment B Sensitivity (No DT_ratio: +Sv, +Ph, +sigma_v)", res_b_no_dt["predictions_df"]),
        ("exp_c", "Experiment C (Formation-Aware: +Stratigraphy, +Hardgrounds)", res_c["predictions_df"]),
    ]

    df_st_long, df_st_wide = build_individual_station_errors(eval_configs, station_comp_df, df_targ)
    _save_dual(df_st_long, exp_metrics_dir / "individual_physical_station_errors.csv", exp_metrics_dir / "individual_physical_station_errors.json")
    _save_dual(df_st_long, exp_tables_dir / "individual_physical_station_errors.csv", exp_tables_dir / "individual_physical_station_errors.json")

    _save_dual(df_st_wide, exp_metrics_dir / "individual_station_benchmark_matrix.csv", exp_metrics_dir / "individual_station_benchmark_matrix.json")
    _save_dual(df_st_wide, exp_tables_dir / "individual_station_benchmark_matrix.csv", exp_tables_dir / "individual_station_benchmark_matrix.json")

    # Compute Depth Sensitivity Profile for Experiment A MLP
    exp_a_scaler = joblib.load(exp_models_dir / "exp_a_scaler.joblib")
    exp_a_mlp = res_a["trained_models"]["mlp"]
    df_mlp_sensitivity = compute_mlp_depth_sensitivity(
        exp_a_mlp, exp_a_scaler, res_a["features"], df_pred, df_targ
    )
    _save_dual(df_mlp_sensitivity, exp_metrics_dir / "mlp_depth_sensitivity_profile.csv", exp_metrics_dir / "mlp_depth_sensitivity_profile.json")
    _save_dual(df_mlp_sensitivity, exp_tables_dir / "mlp_depth_sensitivity_profile.csv", exp_tables_dir / "mlp_depth_sensitivity_profile.json")

    # Generate diagnostic figures
    fig_paths = generate_experiment_figures(
        df_targ,
        baseline_pred_df,
        res_a["predictions_df"],
        res_b["predictions_df"],
        res_c["predictions_df"],
        station_comp_df,
        exp_figures_dir,
        exp_b_no_dt_pred_df=res_b_no_dt["predictions_df"],
    )

    return {
        "exp_a": res_a,
        "exp_b": res_b,
        "exp_b_no_dt": res_b_no_dt,
        "exp_c": res_c,
        "combined_part_metrics": combined_part_metrics,
        "combined_st_benchmarks": combined_st_benchmarks,
        "combined_prov_metrics": combined_prov_metrics,
        "individual_station_errors": df_st_long,
        "individual_station_matrix": df_st_wide,
        "mlp_depth_sensitivity": df_mlp_sensitivity,
        "fig_paths": fig_paths,
    }
