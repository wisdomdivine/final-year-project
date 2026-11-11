"""
Run Controlled Experimental Comparisons:
- Experiment A: Depth-Aware Models (+TVD_MSL)
- Experiment B: Physics-Informed Models (+Sv, +Ph, +sigma_v_hydro, +DT_nct_ratio)
- Experiment C: Formation-Aware Models (+Lithostratigraphy, +Hardgrounds)

Strictly preserves the 70:15:15 contiguous depth split and leaves baseline models untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.pipeline import run_all_experiments


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    print("Executing Well 15/9-F-12 Controlled Experimental Comparisons (A, B, C)...")
    results = run_all_experiments(base_dir)

    st_bench = results["combined_st_benchmarks"]

    print("\n=== EXPERIMENTAL COMPARISON: PHYSICAL STATION BENCHMARKS ===")
    cols_to_print = [
        "experiment_name",
        "model_name",
        "train_15_stations_RMSE_bar",
        "val_test16_pred_bar",
        "val_test16_residual_bar",
        "test20_pred_bar",
        "test21_pred_bar",
        "test20_21_MAE_bar",
    ]
    print(st_bench[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()
