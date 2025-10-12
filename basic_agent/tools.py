from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Any

import pandas as pd


def csv_to_variable(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV path not found: {path_obj}")

    df = pd.read_csv(path_obj)
    df.attrs["source_path"] = str(path_obj)
    return df


def summarize_data(df: pd.DataFrame) -> str:
    """Generate a concise textual summary for a DataFrame."""
    if df.empty:
        return "DataFrame is empty."

    buffer: List[str] = []
    buffer.append(f"Rows: {len(df)}, Columns: {df.shape[1]}")

    column_summary = ", ".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())
    buffer.append(f"Column types: {column_summary}")

    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        desc = numeric_cols.describe().round(3)
        buffer.append("Numeric summary:\n" + desc.to_string())

    categorical_cols = df.select_dtypes(exclude="number")
    if not categorical_cols.empty:
        head_info = categorical_cols.head(3)
        buffer.append("Sample categorical values:\n" + head_info.to_string())

    return "\n\n".join(buffer)


def combine_summarize(dfs: Iterable[pd.DataFrame]) -> str:
    """Concatenate multiple DataFrames and summarize the combined result."""
    df_list = list(dfs)
    if not df_list:
        raise ValueError("No DataFrames provided to combine and summarize")

    combined = pd.concat(df_list, ignore_index=True)
    combined.attrs["combined_from"] = [df.attrs.get("source_path") for df in df_list]
    return summarize_data(combined)


TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "CSV_TO_VARIABLE": csv_to_variable,
    "SUMMARIZE_DATA": summarize_data,
    "COMBINE_SUMMARIZE": combine_summarize,
}
