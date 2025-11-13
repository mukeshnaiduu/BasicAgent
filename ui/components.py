from __future__ import annotations

import json
from ast import literal_eval
from typing import Any, Dict, Optional

import streamlit as st


def render_node(step: Dict[str, Any], result: Any, index: int, raw_output: Optional[str] = None) -> None:
    """Render a workflow node with parameters and output preview."""
    title = step.get("description") or step.get("tool") or f"Step {index}"
    container = st.container()
    with container:
        st.markdown(f"#### Step {index}: {title}")
        st.caption(f"Tool: {step.get('tool', 'unknown')} · ID: {step.get('id', 'n/a')}")

        params = step.get("params") or {}
        if params:
            with st.expander("Parameters", expanded=False):
                st.json(params)

        # Normalize string results so that common "None"/empty outputs don't render as raw text
        normalized = result
        if isinstance(result, str):
            stripped = result.strip()
            if stripped.lower() in {"", "none", "null"}:
                normalized = None
            else:
                parsed = None
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    try:
                        parsed = literal_eval(stripped)
                    except Exception:
                        parsed = None
                if isinstance(parsed, (dict, list)):
                    normalized = parsed
        else:
            stripped = None  # ensure defined for later use

        if not normalized:
            st.info("No output produced for this step.")
            return

        if isinstance(normalized, dict):
            result_dict = normalized
            kind = result_dict.get("type")
            if kind == "dataframe":
                shape = result_dict.get("shape")
                if shape:
                    st.markdown(f"Shape: {shape[0]} rows × {shape[1]} columns")
                preview = result_dict.get("preview")
                if preview is not None:
                    st.dataframe(preview, use_container_width=True)
            elif kind == "dict":
                st.json(result_dict.get("content", {}))
            elif kind == "text":
                st.write(result_dict.get("content", ""))
            else:
                st.write(result_dict)
        else:
            st.write(normalized)

        if raw_output:
            with st.expander("Raw output", expanded=False):
                st.code(raw_output)


def render_summary(text: str) -> None:
    st.subheader("Summary")
    st.code(text)
 
