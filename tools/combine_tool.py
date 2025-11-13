from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

import pandas as pd


TOOL_INFO = {
    "name": "combine",
    "purpose": "Join or merge multiple DataFrames into one using specified keys and join types.",
    "parameters": {
        "dfs": {"type": "list[DataFrame|id]", "required": True, "description": "Ordered list of DataFrames or their identifiers to combine (left-to-right).", "example": ["df_orders", "df_customers"]},
        "on": {"type": "list[string]", "required": False, "description": "Columns to join on. If omitted, common column intersection is used when possible.", "example": ["customer_id"]},
        "how": {"type": "string", "required": False, "description": "Join type: inner|left|right|outer", "example": "inner"},
        "right_index": {"type": "boolean", "required": False, "description": "If true, join on the right index instead of columns.", "example": False},
    },
    "output": {"type": "DataFrame", "description": "Combined pandas.DataFrame"},
}


def combine_dfs(dfs: Iterable[pd.DataFrame], on: Optional[List[str]] = None, how: str = "inner") -> pd.DataFrame:
    """Combine DataFrames sequentially using the provided keys.

    Args:
        dfs: iterable of DataFrames. When multiple frames are provided they are
            merged left-to-right: result = df1.merge(df2).merge(df3)...
        on: list of column names to join on. If None and frames share common
            columns, uses the intersection of column names.
        how: join type (inner, left, right, outer)

    Returns:
        Combined DataFrame.

    Notes:
        - This function performs straightforward pandas merges and will raise
          if required columns are missing. The caller (agent) should validate
          the join plan first.
    """
    df_list = list(dfs)
    if not df_list:
        return pd.DataFrame()

    result = df_list[0].copy()
    for right in df_list[1:]:
        if on is None:
            common = [c for c in result.columns if c in right.columns]
            if not common:
                raise ValueError("No common columns to join on; please provide 'on' parameter")
            join_on = common
        else:
            join_on = on
        result = result.merge(right, how=how, on=join_on)
    result.attrs["combined_from"] = [df.attrs.get("source_path") for df in df_list]
    return result
