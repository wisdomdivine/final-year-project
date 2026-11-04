"""
EDA, Multicollinearity, and Partition Distribution Analysis
for Well 15/9-F-12 (Volve Field).

This script:
1. Loads the reconstructed 1,811-row predictor dataset.
2. Computes the Pearson correlation matrix.
3. Calculates Variance Inflation Factors (VIF) to detect multicollinearity.
4. Breaks down the 70:15:15 depth-based partitions (Training, Validation, Testing)
   and evaluates distribution metrics (mean, std, min, median, max) across intervals.
5. Saves all generated summary tables to CSV and JSON in the tables/ directory.
"""

import json
import os
import numpy as np
import pandas as pd


def compute_vif_numpy(X):
    """Compute VIF for each column in matrix X with intercept."""
    # Add intercept column
    n_samples, n_features = X.shape
    c = np.ones((n_samples, 1))
    X_with_const = np.hstack([c, X])
    
    # Correlation/covariance inversion approach
    # For standardized features, inv(corr) diagonal gives VIF
    corr = np.corrcoef(X, rowvar=False)
    inv_corr = np.linalg.inv(corr)
    vifs = np.diag(inv_corr)
    return vifs


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "F12_predictor_dataset_1811.csv")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Shape: {df.shape}")

    feature_cols = [col for col in df.columns]
    predictor_cols = [col for col in df.columns if col != "DEPTH"]

    # 1. Pearson Correlation Matrix
    corr_matrix = df.corr(method="pearson").round(4)
    corr_csv_path = os.path.join(tables_dir, "pearson_correlation_matrix.csv")
    corr_json_path = os.path.join(tables_dir, "pearson_correlation_matrix.json")
    corr_matrix.to_csv(corr_csv_path)
    corr_matrix.to_json(corr_json_path, orient="index", indent=2)
    print("Saved Pearson correlation matrix.")

    # 2. Multicollinearity (VIF)
    vifs = compute_vif_numpy(df[feature_cols].values)
    vif_records = []
    for col, vif_val in zip(feature_cols, vifs):
        status = "Severe" if vif_val > 10 else ("Moderate" if vif_val > 5 else "Low")
        vif_records.append({
            "variable": col,
            "vif": round(float(vif_val), 2),
            "multicollinearity_level": status
        })

    vif_df = pd.DataFrame(vif_records).sort_values(by="vif", ascending=False)
    vif_csv_path = os.path.join(tables_dir, "variance_inflation_factors.csv")
    vif_json_path = os.path.join(tables_dir, "variance_inflation_factors.json")
    vif_df.to_csv(vif_csv_path, index=False)
    with open(vif_json_path, "w") as f:
        json.dump(vif_df.to_dict(orient="records"), f, indent=2)
    print("Saved Variance Inflation Factors.")

    # 3. Depth-based partition breakdown (70:15:15)
    train_n = 1268
    val_n = 272
    test_n = 271

    df_train = df.iloc[:train_n]
    df_val = df.iloc[train_n:train_n + val_n]
    df_test = df.iloc[train_n + val_n:]

    partition_stats = []
    for col in feature_cols:
        stat_item = {
            "variable": col,
            "overall_mean": round(float(df[col].mean()), 4),
            "train_mean": round(float(df_train[col].mean()), 4),
            "val_mean": round(float(df_val[col].mean()), 4),
            "test_mean": round(float(df_test[col].mean()), 4),
            "train_std": round(float(df_train[col].std()), 4),
            "val_std": round(float(df_val[col].std()), 4),
            "test_std": round(float(df_test[col].std()), 4),
            "train_min": round(float(df_train[col].min()), 4),
            "val_min": round(float(df_val[col].min()), 4),
            "test_min": round(float(df_test[col].min()), 4),
            "train_max": round(float(df_train[col].max()), 4),
            "val_max": round(float(df_val[col].max()), 4),
            "test_max": round(float(df_test[col].max()), 4)
        }
        partition_stats.append(stat_item)

    part_df = pd.DataFrame(partition_stats)
    part_csv_path = os.path.join(tables_dir, "partition_distribution_comparison.csv")
    part_json_path = os.path.join(tables_dir, "partition_distribution_comparison.json")
    part_df.to_csv(part_csv_path, index=False)
    with open(part_json_path, "w") as f:
        json.dump(partition_stats, f, indent=2)
    print("Saved Partition Distribution Comparison.")

    print("\n=== TOP MULTICOLLINEARITY (VIF) ===")
    print(vif_df.to_string(index=False))

    print("\n=== DEPTH COVERAGE BY PARTITION ===")
    print(f"Training:   {df_train['DEPTH'].min():.2f}m - {df_train['DEPTH'].max():.2f}m ({len(df_train)} rows)")
    print(f"Validation: {df_val['DEPTH'].min():.2f}m - {df_val['DEPTH'].max():.2f}m ({len(df_val)} rows)")
    print(f"Testing:    {df_test['DEPTH'].min():.2f}m - {df_test['DEPTH'].max():.2f}m ({len(df_test)} rows)")


if __name__ == "__main__":
    main()
