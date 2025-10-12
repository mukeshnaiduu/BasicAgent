from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basic_agent.tools import combine_summarize, summarize_data


def test_summarize_data_includes_shape_information() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = summarize_data(df)
    assert "Rows: 3" in summary
    assert "Column types" in summary


def test_combine_summarize_handles_multiple_frames() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
    summary = combine_summarize([df1, df2])
    assert "Rows: 4" in summary
    assert "Numeric summary" in summary
