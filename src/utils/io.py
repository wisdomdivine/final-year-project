"""I/O utilities for tabular data persistence."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating parent directories if needed."""
    p = Path(path)
    # If path looks like a directory (no suffix and doesn't exist yet, or existing dir), ensure it
    target_dir = p.parent if p.suffix else p
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_dataframe_dual(
    df: pd.DataFrame,
    base_path_without_ext: Union[str, Path],
    orient: str = "records",
    indent: int = 2,
    index: bool = False,
) -> Dict[str, str]:
    """Save a DataFrame to both CSV and JSON formats simultaneously."""
    base_path = Path(base_path_without_ext)
    ensure_directory(base_path.parent)

    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")

    df.to_csv(csv_path, index=index)
    df.to_json(json_path, orient=orient, indent=indent)

    return {"csv": str(csv_path), "json": str(json_path)}


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> str:
    """Save serializable data to JSON file."""
    p = Path(path)
    ensure_directory(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return str(p)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
