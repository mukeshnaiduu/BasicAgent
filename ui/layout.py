from __future__ import annotations

import streamlit as st


def page_setup(title: str = "BasicAgent") -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)


def two_column_layout(left_ratio: int = 3, right_ratio: int = 7):
    total = left_ratio + right_ratio
    left_width = int(100 * left_ratio / total)
    right_width = 100 - left_width
    return st.columns([left_ratio, right_ratio])
 
