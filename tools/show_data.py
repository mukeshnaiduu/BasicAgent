from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


TOOL_INFO = {
    "name": "show_data",
    "purpose": "Return a small preview (head) of a DataFrame for quick inspection.",
    "parameters": {
        "df": {"type": "DataFrame|id", "required": True, "description": "The target DataFrame or its identifier (recommended).", "example": "df_customers or pass DataFrame object"},
        "rows": {"type": "integer", "required": False, "description": "Number of rows to return.", "example": 10},
        "display_name": {"type": "string", "required": False, "description": "Optional display name to show in UI.", "example": "customers.csv"}
    },
    "output": {"type": "object", "description": "dict with display_name, shape, rows, preview (pandas.DataFrame head)"},
}


def preview_df(df: pd.DataFrame, rows: int = 10, display_name: Optional[str] = None) -> Dict[str, Any]:
    preview_df = df.head(int(rows))
    return {
        "display_name": display_name or df.attrs.get("display_name") or df.attrs.get("source_path", ""),
        "shape": df.shape,
        "rows": int(rows),
        "preview": preview_df,
    }