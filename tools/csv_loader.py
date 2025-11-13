from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd


TOOL_INFO = {
    "name": "csv_loader",
    "purpose": "Load CSV files into pandas DataFrames (from upload or disk).",
    "parameters": {
        "path": {"type": "string", "required": False, "description": "Filesystem path to CSV file.", "example": "data/customers.csv"},
        "uploaded": {"type": "file", "required": False, "description": "Uploaded file object (Streamlit UploadedFile). Use when user uploaded a file.", "example": "<uploaded file>"},
        "name": {"type": "string", "required": False, "description": "Optional synthetic name to use as source_path for provenance.", "example": "customers.csv"},
    },
    "output": {"type": "DataFrame", "description": "pandas.DataFrame with attrs['source_path'] set for provenance"},
}


def load_from_path(path: str | Path, name: Optional[str] = None, **read_kwargs: Any) -> pd.DataFrame:
    """Load a CSV from disk and return a DataFrame.

    Raises FileNotFoundError when the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p, **read_kwargs)
    source = name or str(p)
    df.attrs["source_path"] = source
    return df


def load_from_uploaded(uploaded: Any, name: Optional[str] = None, **read_kwargs: Any) -> pd.DataFrame:
    """Load from an uploaded file-like object and set a synthetic source name.

    The caller is expected to pass the Streamlit `UploadedFile` or similar.
    """
    df = pd.read_csv(uploaded, **read_kwargs)
    synthetic = name or getattr(uploaded, "name", "uploaded://memory")
    df.attrs["source_path"] = str(synthetic)
    return df
