"""
Execute the complete Model Training, Evaluation, and Diagnostics Pipeline
for Well 15/9-F-12 (Volve Field).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.pipeline import run_training_pipeline


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    print("Executing Well 15/9-F-12 model training and evaluation pipeline...")
    results = run_training_pipeline(base_dir)

    part_df = results["eval_tables"]["partition_metrics"]
    st_df = results["eval_tables"]["station_benchmarks"]

    print("\n=== PARTITION METRICS SUMMARY ===")
    print(
        part_df[
            [
                "model_name",
                "partition_subset",
                "n_valid",
                "MAE_bar",
                "RMSE_bar",
                "R2",
                "mean_residual_bar",
            ]
        ].to_string(index=False)
    )

    print("\n=== DISCRETE PHYSICAL STATION BENCHMARKS ===")
    print(
        st_df[
            [
                "model_name",
                "train_15_stations_RMSE_bar",
                "val_test16_pred_bar",
                "val_test16_residual_bar",
                "test20_pred_bar",
                "test21_pred_bar",
                "test20_21_MAE_bar",
                "test20_21_RMSE_bar",
                "excluded_test19_diff_bar",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

