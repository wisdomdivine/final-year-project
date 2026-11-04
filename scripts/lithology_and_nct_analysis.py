"""
Lithology Identification and Normal Compaction Trend (NCT) Analysis
for Well 15/9-F-12 (Volve Field).

This script:
1. Classifies the 1,811 observations into Sandstone vs Shale using the official
   Equinor petrophysical interpretation flags (SAND_FLAG) and petrophysical thresholds (GR, VCL).
2. Fits Normal Compaction Trends (NCT) across shale intervals in the overlying column
   (1,500 m to 3,000 m) using different Gamma Ray thresholds (GR >= 60, GR >= 75, GR >= 90).
3. Evaluates normal compaction transit times (DT_n) across the study interval.
4. Saves structured summary tables to tables/ in CSV and JSON formats.
"""

import json
import os
import numpy as np
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    las_path = os.path.join(base_dir, "data-raw", "WLC_PETRO_COMPUTED_INPUT_1.LAS")
    dataset_1811_path = os.path.join(base_dir, "dataset", "F12_predictor_dataset_1811.csv")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Parse LAS data to extract flags
    with open(las_path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    data_start = [i for i, l in enumerate(lines) if l.strip().startswith("~A")][0] + 1
    cols = ["DEPTH", "CARB_FLAG", "COAL_FLAG", "DT", "GR", "NPHI", "RD", "RHOB", "ROP5_RM", "RS", "RT", "SAND_FLAG"]
    data = [[float(p) for p in l.strip().split()] for l in lines[data_start:] if len(l.strip().split()) == len(cols)]
    df_las = pd.DataFrame(data, columns=cols).replace(-999.25, np.nan)

    # 2. Match with 1,811 predictor dataset
    df_1811 = pd.read_csv(dataset_1811_path)
    study_las = df_las[(df_las["DEPTH"] >= 3127.70) & (df_las["DEPTH"] <= 3403.56)].copy()
    merged = pd.merge(df_1811, study_las[["DEPTH", "SAND_FLAG", "CARB_FLAG", "COAL_FLAG"]], on="DEPTH")

    # Lithology breakdown
    shale_mask = (merged["SAND_FLAG"] == 0.0)
    sand_mask = (merged["SAND_FLAG"] == 1.0)

    litho_summary = [
        {
            "lithology": "Sandstone (Reservoir Sand)",
            "flag_definition": "SAND_FLAG == 1.0",
            "row_count": int(sand_mask.sum()),
            "percentage": round(float(sand_mask.sum() / len(merged) * 100), 2),
            "gr_mean_api": round(float(merged.loc[sand_mask, "GR"].mean()), 2),
            "vcl_mean": round(float(merged.loc[sand_mask, "VCL"].mean()), 4),
            "dt_mean_us_ft": round(float(merged.loc[sand_mask, "DT"].mean()), 2),
            "rhob_mean_g_cm3": round(float(merged.loc[sand_mask, "RHOB"].mean()), 4),
            "k_mean_md": round(float(merged.loc[sand_mask, "K"].mean()), 2),
            "phif_mean": round(float(merged.loc[sand_mask, "PHIF"].mean()), 4)
        },
        {
            "lithology": "Shale / Non-Sand (Caprock & Intra-reservoir)",
            "flag_definition": "SAND_FLAG == 0.0",
            "row_count": int(shale_mask.sum()),
            "percentage": round(float(shale_mask.sum() / len(merged) * 100), 2),
            "gr_mean_api": round(float(merged.loc[shale_mask, "GR"].mean()), 2),
            "vcl_mean": round(float(merged.loc[shale_mask, "VCL"].mean()), 4),
            "dt_mean_us_ft": round(float(merged.loc[shale_mask, "DT"].mean()), 2),
            "rhob_mean_g_cm3": round(float(merged.loc[shale_mask, "RHOB"].mean()), 4),
            "k_mean_md": round(float(merged.loc[shale_mask, "K"].mean()), 2),
            "phif_mean": round(float(merged.loc[shale_mask, "PHIF"].mean()), 4)
        }
    ]

    litho_df = pd.DataFrame(litho_summary)
    litho_df.to_csv(os.path.join(tables_dir, "lithology_classification.csv"), index=False)
    with open(os.path.join(tables_dir, "lithology_classification.json"), "w") as f:
        json.dump(litho_summary, f, indent=2)
    print("Saved lithology classification table.")

    # 3. Normal Compaction Trend (NCT) comparison across GR thresholds
    nct_results = []
    thresholds = [60.0, 75.0, 90.0]

    for gr_thresh in thresholds:
        shale_subset = df_las[
            (df_las["DT"].notnull()) &
            (df_las["GR"] >= gr_thresh) &
            (df_las["DT"] > 40) &
            (df_las["DT"] < 160)
        ].copy()

        norm_interval = shale_subset[(shale_subset["DEPTH"] >= 1500) & (shale_subset["DEPTH"] <= 3000)]
        slope, intercept = np.polyfit(norm_interval["DEPTH"], np.log(norm_interval["DT"]), 1)
        dt_surface = float(np.exp(intercept))
        compaction_coeff = float(-slope)

        dt_top = float(np.exp(intercept + slope * 3127.71))
        dt_mid = float(np.exp(intercept + slope * 3265.63))
        dt_base = float(np.exp(intercept + slope * 3403.55))

        nct_results.append({
            "gr_threshold_api": gr_thresh,
            "sample_count": len(shale_subset),
            "norm_interval_samples": len(norm_interval),
            "dt_surface_us_ft": round(dt_surface, 2),
            "compaction_coefficient_per_m": round(compaction_coeff, 6),
            "normal_dt_top_3128m": round(dt_top, 2),
            "normal_dt_mid_3266m": round(dt_mid, 2),
            "normal_dt_base_3404m": round(dt_base, 2)
        })

    nct_df = pd.DataFrame(nct_results)
    nct_df.to_csv(os.path.join(tables_dir, "nct_trend_parameters.csv"), index=False)
    with open(os.path.join(tables_dir, "nct_trend_parameters.json"), "w") as f:
        json.dump(nct_results, f, indent=2)
    print("Saved NCT trend parameters table.")

    print("\n=== LITHOLOGY SUMMARY ===")
    print(litho_df[["lithology", "row_count", "percentage", "gr_mean_api", "vcl_mean", "dt_mean_us_ft", "rhob_mean_g_cm3"]].to_string(index=False))

    print("\n=== NCT TREND PARAMETERS ===")
    print(nct_df.to_string(index=False))


if __name__ == "__main__":
    main()

