"""
Parse MWD Stethoscope Formation Pressure Tests (Schlumberger)
from official StatoilHydro / Equinor 15/9-F-12 records.

Saves structured table to CSV and JSON formats in tables/.
"""

import json
import os
import re
import pandas as pd


def parse_pressure_asc(asc_path):
    records = []
    with open(asc_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    start_parsing = False
    for line in lines:
        if line.strip().startswith("TEST FRM"):
            continue
        if line.strip().startswith("NO"):
            start_parsing = True
            continue
        if not start_parsing or not line.strip():
            continue
        if line.strip().startswith("*"):
            break

        # Fixed width / regex parsing
        # Pattern matches:
        # TEST NO (int), FRM (str), DEPTH.MD, DEPTH.TVD(RKB), DEPTH.TVD(MSL), PRES.HDRB, PRES.FM, PRES.HDRA, TEMP, MOBL, QUAL, REM
        parts = line.strip().split()
        if not parts or not parts[0].isdigit():
            continue

        test_no = int(parts[0])
        formation = parts[1]

        # Handle station 19 which had a shutdown and missing TVD/pressure
        if test_no == 19:
            records.append({
                "test_no": test_no,
                "formation": formation,
                "depth_md_m": 3403.5,
                "depth_tvd_rkb_m": None,
                "depth_tvd_msl_m": -54.9,
                "pres_hdrb_bar": None,
                "pres_fm_bar": None,
                "pres_hdra_bar": None,
                "temp_degc": None,
                "mobility_cmd_cp": None,
                "quality_code": 0,
                "quality_description": "Lost seal or tool shutdown",
                "remarks": "Unit shutdown. Memory results.",
                "in_study_interval": True,
                "valid_pressure_test": False
            })
            continue

        depth_md = float(parts[2])
        depth_tvd_rkb = float(parts[3])
        depth_tvd_msl = float(parts[4])
        pres_hdrb = float(parts[5])
        pres_fm = float(parts[6])
        pres_hdra = float(parts[7])
        temp = float(parts[8])
        mobl = float(parts[9])
        qual = int(parts[10])
        remarks = " ".join(parts[11:])

        qual_desc_map = {
            0: "Lost seal / unstabilized",
            1: "Tight formation / dry test",
            2: "Poor permeability (< 10 mD)",
            3: "Good permeability (10-100 mD)",
            4: "Very good permeability (> 100 mD)"
        }

        # Valid test criteria: qual >= 2 and pressure above minimum threshold (filtering dry test 69.74 bar)
        is_valid = (qual >= 2) and (pres_fm > 200.0)
        in_interval = (3127.71 <= depth_md <= 3403.55)

        records.append({
            "test_no": test_no,
            "formation": formation,
            "depth_md_m": depth_md,
            "depth_tvd_rkb_m": depth_tvd_rkb,
            "depth_tvd_msl_m": depth_tvd_msl,
            "pres_hdrb_bar": pres_hdrb,
            "pres_fm_bar": pres_fm,
            "pres_hdra_bar": pres_hdra,
            "temp_degc": temp,
            "mobility_cmd_cp": mobl,
            "quality_code": qual,
            "quality_description": qual_desc_map.get(qual, "Unknown"),
            "remarks": remarks,
            "in_study_interval": in_interval,
            "valid_pressure_test": is_valid
        })

    return records


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asc_path = os.path.join(base_dir, "data-raw", "FM_PRESS_COMPUTED_MWD_1.ASC")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    records = parse_pressure_asc(asc_path)
    df = pd.DataFrame(records)

    csv_path = os.path.join(tables_dir, "mwd_formation_pressure_tests.csv")
    json_path = os.path.join(tables_dir, "mwd_formation_pressure_tests.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Parsed {len(records)} pressure test records.")
    print(f"Saved to: {csv_path} and {json_path}")

    # Summary breakdown
    in_interval_df = df[df["in_study_interval"]]
    valid_interval_df = in_interval_df[in_interval_df["valid_pressure_test"]]
    print(f"\nTotal stations: {len(df)}")
    print(f"Stations within 3127.71m - 3403.55m interval: {len(in_interval_df)}")
    print(f"Valid stabilized stations within interval: {len(valid_interval_df)}")
    print("\nValid tests in study interval:")
    print(valid_interval_df[["test_no", "formation", "depth_md_m", "depth_tvd_rkb_m", "pres_fm_bar", "quality_code"]].to_string(index=False))


if __name__ == "__main__":
    main()

