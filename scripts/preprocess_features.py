"""
Feature Preprocessing and Transformation Pipeline
for Well 15/9-F-12 (Volve Field).

This script:
1. Implements the strict 70:15:15 contiguous depth split:
   - Training:   3127.7052 m to 3320.7960 m (1,268 rows)
   - Validation: 3320.9484 m to 3362.2488 m (272 rows)
   - Testing:    3362.4012 m to 3403.5492 m (271 rows)
2. Generates feature sets tailored for each model architecture:
   - Native unscaled features (for Random Forest and CatBoost)
   - Standard-scaled features (fitted strictly on training set, for MLP)
   - Log10-transformed + scaled features (addressing skew in K, RS, RD, RT, for MLR and MLP)
3. Outputs transformation statistics and saves preprocessed arrays to disk.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "F12_predictor_dataset_1811.csv")
    tables_dir = os.path.join(base_dir, "tables")
    processed_dir = os.path.join(base_dir, "dataset", "processed")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 1. Depth-based contiguous partitioning
    train_n = 1268
    val_n = 272
    test_n = 271

    df_train = df.iloc[:train_n].copy()
    df_val = df.iloc[train_n:train_n + val_n].copy()
    df_test = df.iloc[train_n + val_n:].copy()

    # 2. Skewness analysis and log10 transformation
    skewed_cols = ["RS", "K", "RD", "RT"]
    trans_records = []

    for col in df.columns:
        raw_skew = float(df[col].skew())
        if col in skewed_cols:
            log_series = np.log10(df[col])
            log_skew = float(log_series.skew())
            trans_records.append({
                "variable": col,
                "raw_skewness": round(raw_skew, 4),
                "transformed_skewness": round(log_skew, 4),
                "transformation": "log10",
                "recommended_for": "MLR, MLP"
            })
        else:
            trans_records.append({
                "variable": col,
                "raw_skewness": round(raw_skew, 4),
                "transformed_skewness": round(raw_skew, 4),
                "transformation": "none",
                "recommended_for": "All models"
            })

    trans_df = pd.DataFrame(trans_records)
    trans_csv = os.path.join(tables_dir, "feature_transformations.csv")
    trans_json = os.path.join(tables_dir, "feature_transformations.json")
    trans_df.to_csv(trans_csv, index=False)
    with open(trans_json, "w") as f:
        json.dump(trans_records, f, indent=2)
    print("Saved feature transformation summary.")

    # 3. Create preprocessed datasets

    # Set A: Native unscaled (for Random Forest and CatBoost)
    df_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    df_val.to_csv(os.path.join(processed_dir, "X_val_raw.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    # Set B: Standard Scaled (fitted strictly on training set)
    scaler_std = StandardScaler()
    scaler_std.fit(df_train)

    X_train_scaled = pd.DataFrame(scaler_std.transform(df_train), columns=df.columns)
    X_val_scaled = pd.DataFrame(scaler_std.transform(df_val), columns=df.columns)
    X_test_scaled = pd.DataFrame(scaler_std.transform(df_test), columns=df.columns)

    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train_scaled.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(processed_dir, "X_val_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test_scaled.csv"), index=False)

    # Set C: Log-transformed + Scaled
    df_train_log = df_train.copy()
    df_val_log = df_val.copy()
    df_test_log = df_test.copy()

    for col in skewed_cols:
        df_train_log[col] = np.log10(df_train_log[col])
        df_val_log[col] = np.log10(df_val_log[col])
        df_test_log[col] = np.log10(df_test_log[col])

    scaler_log = StandardScaler()
    scaler_log.fit(df_train_log)

    X_train_log_scaled = pd.DataFrame(scaler_log.transform(df_train_log), columns=df.columns)
    X_val_log_scaled = pd.DataFrame(scaler_log.transform(df_val_log), columns=df.columns)
    X_test_log_scaled = pd.DataFrame(scaler_log.transform(df_test_log), columns=df.columns)

    X_train_log_scaled.to_csv(os.path.join(processed_dir, "X_train_log_scaled.csv"), index=False)
    X_val_log_scaled.to_csv(os.path.join(processed_dir, "X_val_log_scaled.csv"), index=False)
    X_test_log_scaled.to_csv(os.path.join(processed_dir, "X_test_log_scaled.csv"), index=False)

    # 4. Save scaler parameters for full reproducibility and data leakage prevention audit
    scaler_audit = {
        "mean_training_values": dict(zip(df.columns, np.round(scaler_std.mean_, 4).tolist())),
        "std_training_values": dict(zip(df.columns, np.round(scaler_std.scale_, 4).tolist())),
        "mean_training_log_values": dict(zip(df.columns, np.round(scaler_log.mean_, 4).tolist())),
        "std_training_log_values": dict(zip(df.columns, np.round(scaler_log.scale_, 4).tolist())),
    }
    with open(os.path.join(tables_dir, "scaler_audit.json"), "w") as f:
        json.dump(scaler_audit, f, indent=2)

    print("Preprocessed feature sets saved to dataset/processed/ successfully.")
    print("Scaler audit saved to tables/scaler_audit.json.")


if __name__ == "__main__":
    main()
