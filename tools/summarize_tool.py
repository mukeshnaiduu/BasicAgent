from __future__ import annotations

from typing import Iterable, List, Dict, Any

import pandas as pd


TOOL_INFO = {
    "name": "summarize",
    "purpose": "Generate a concise textual summary for a single DataFrame.",
    "parameters": {
        "df": {"type": "DataFrame|id", "required": True, "description": "Target DataFrame or its identifier.", "example": "df_customers"},
        "numeric_only": {"type": "boolean", "required": False, "description": "Include numeric descriptive stats if true (default true).", "example": True}
    },
    "output": {"type": "string", "description": "Short text summary describing rows, columns, types, and numeric stats"},
}


def summarize_df(df: pd.DataFrame, numeric_only: bool = True) -> str:
    """Return a short textual summary of the DataFrame.

    Only summary text is returned (no extra metadata). Keep output concise so
    it can be embedded directly into LLM prompts.
    """
    if df.empty:
        return "DataFrame is empty."
    parts: List[str] = []
    parts.append(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    dtypes = ", ".join(f"{c}:{t}" for c, t in df.dtypes.items())
    parts.append(f"Column types: {dtypes}")
    if numeric_only:
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            desc = numeric.describe().round(3)
            parts.append("Numeric summary:\n" + desc.to_string())
    return "\n\n".join(parts)

