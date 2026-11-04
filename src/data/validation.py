"""Data validation module ensuring physical realism and data integrity."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


EXPECTED_ROW_COUNT = 1811
EXPECTED_DEPTH_START = 3127.7052
EXPECTED_DEPTH_END = 3403.5492
EXPECTED_STEP = 0.1524

EXPECTED_PREDICTORS = [
    "DT", "GR", "NPHI", "RD", "RHOB", "RS", "RT",
    "ROP", "BVW", "K", "PHIF", "SW", "VCL"
]
DEPTH_COL = "DEPTH"
EXPECTED_COLUMNS = [DEPTH_COL] + EXPECTED_PREDICTORS


def validate_predictor_dataset(
    df: pd.DataFrame,
    depth_col: str = DEPTH_COL,
) -> Dict[str, Any]:
    """
    Run comprehensive validation checks on the 1,811-observation predictor dataset.
    
    Returns a dictionary summarizing all validation results.
    """
    results: Dict[str, Any] = {
        "status": "PASS",
        "checks": {},
        "summary": {},
        "failures": [],
    }

    # 1. Shape check
    actual_rows = len(df)
    shape_pass = actual_rows == EXPECTED_ROW_COUNT
    results["checks"]["row_count"] = {
        "expected": EXPECTED_ROW_COUNT,
        "actual": actual_rows,
        "pass": shape_pass,
    }
    if not shape_pass:
        results["failures"].append(
            f"Row count mismatch: expected {EXPECTED_ROW_COUNT}, got {actual_rows}"
        )

    # 2. Columns check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    cols_pass = len(missing_cols) == 0
    results["checks"]["columns"] = {
        "expected_columns": EXPECTED_COLUMNS,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "pass": cols_pass,
    }
    if not cols_pass:
        results["failures"].append(f"Missing required columns: {missing_cols}")

    # 3. Missing values check
    cols_to_check = [col for col in EXPECTED_COLUMNS if col in df.columns]
    null_counts = df[cols_to_check].isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    nulls_pass = total_nulls == 0
    results["checks"]["missing_values"] = {
        "total_nulls": int(total_nulls),
        "column_nulls": {k: int(v) for k, v in null_counts.items() if v > 0},
        "pass": nulls_pass,
    }
    if not nulls_pass:
        results["failures"].append(f"Dataset contains {total_nulls} null values")

    # 4. Infinite values check
    inf_counts = {
        col: int(np.isinf(df[col]).sum())
        for col in cols_to_check
        if np.isinf(df[col]).sum() > 0
    }
    infs_pass = len(inf_counts) == 0
    results["checks"]["infinite_values"] = {
        "infinite_counts": inf_counts,
        "pass": infs_pass,
    }
    if not infs_pass:
        results["failures"].append(f"Infinite values found: {inf_counts}")

    # 5. Depth monotonicity check
    if depth_col in df.columns:
        depths = df[depth_col].values
        depth_diffs = np.diff(depths)
        is_strictly_monotonic = bool(np.all(depth_diffs > 0))
        min_step = float(np.min(depth_diffs))
        max_step = float(np.max(depth_diffs))
        median_step = float(np.median(depth_diffs))
        step_tolerance = 1e-4

        step_pass = bool(abs(median_step - EXPECTED_STEP) < step_tolerance)
        monotonic_pass = is_strictly_monotonic and step_pass

        results["checks"]["depth_continuity"] = {
            "depth_col": depth_col,
            "depth_min": float(depths[0]),
            "depth_max": float(depths[-1]),
            "min_step_m": min_step,
            "max_step_m": max_step,
            "median_step_m": median_step,
            "strictly_increasing": is_strictly_monotonic,
            "pass": monotonic_pass,
        }
        if not monotonic_pass:
            results["failures"].append(
                f"Depth continuity failed: strictly_increasing={is_strictly_monotonic}, median_step={median_step}"
            )
    else:
        results["failures"].append(f"Depth column '{depth_col}' not found.")
        results["checks"]["depth_continuity"] = {"pass": False}

    # 6. Physical boundary sanity checks
    bounds = {
        "DT": (40.0, 160.0),
        "GR": (0.0, 250.0),
        "NPHI": (-0.05, 0.65),
        "RD": (0.01, 5000.0),
        "RHOB": (1.8, 3.2),
        "RS": (0.01, 10000.0),
        "RT": (0.01, 5000.0),
        "ROP": (0.0, 150.0),
        "BVW": (0.0, 1.0),
        "K": (0.0, 20000.0),
        "PHIF": (0.0, 0.5),
        "SW": (0.0, 1.0),
        "VCL": (0.0, 1.0),
    }
    boundary_violations = {}
    for col, (b_min, b_max) in bounds.items():
        if col in df.columns:
            c_min = float(df[col].min())
            c_max = float(df[col].max())
            if c_min < b_min or c_max > b_max:
                boundary_violations[col] = {
                    "expected_bounds": [b_min, b_max],
                    "actual_range": [c_min, c_max],
                }

    bounds_pass = len(boundary_violations) == 0
    results["checks"]["physical_boundaries"] = {
        "violations": boundary_violations,
        "pass": bounds_pass,
    }
    if not bounds_pass:
        results["failures"].append(f"Physical boundary violations: {boundary_violations}")

    # Set overall status
    if len(results["failures"]) > 0:
        results["status"] = "FAIL"

    # Compute summary statistics
    desc_cols = [c for c in EXPECTED_PREDICTORS if c in df.columns]
    desc = df[desc_cols].describe().T
    desc["skewness"] = df[desc_cols].skew()
    results["summary"] = desc.to_dict(orient="index")

    return results
