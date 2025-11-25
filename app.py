from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# load environment variables from .env so GEMINI_API_KEY (and others) are available
load_dotenv(dotenv_path=Path('.env'))

from core.agent import Agent, simple_llm_client_from_gemini, simple_llm_client_from_ollama
from core.llm_manager import LLMManager
from configs.settings import load_settings
from ui.components import render_node, render_summary
from ui.layout import page_setup, two_column_layout


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    ts = int(time.time() * 1000)
    filename = f"{ts}_{uploaded_file.name}"
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def cleanup_uploads() -> None:
    # Remove uploaded files after task completion
    try:
        shutil.rmtree(UPLOAD_DIR)
    except Exception:
        pass
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
def main() -> None:
    page_setup("BasicAgent Streamlit")
    settings = load_settings()

    left_col, right_col = two_column_layout()

    with left_col:
        st.header("Inputs")
        provider_options = [f"Ollama ({settings.ollama_model})", f"Gemini ({settings.gemini_model})"]
        # Prefer Gemini as the default when the API key is configured to avoid
        # requiring a local Ollama server unless the user explicitly picks it.
        has_gemini = bool(settings.gemini_api_key)
        default_provider = provider_options[1] if has_gemini else provider_options[0]
        if "llm_provider" not in st.session_state:
            st.session_state.llm_provider = default_provider
        provider_index = provider_options.index(st.session_state.llm_provider) if st.session_state.llm_provider in provider_options else provider_options.index(default_provider)
        provider = st.selectbox("LLM provider", provider_options, index=provider_index, key="llm_provider_select")
        st.session_state.llm_provider = provider
        if provider.startswith("Ollama"):
            st.caption("Ensure the Ollama service is running (e.g., `ollama run llama3.1`).")
        uploaded = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])  # user uploads
        if "uploaded_paths" not in st.session_state:
            st.session_state.uploaded_paths = {}

        if uploaded:
            for up in uploaded:
                path = save_uploaded_file(up)
                st.session_state.uploaded_paths[up.name] = path

        st.markdown("**Uploaded files**")
        for name, p in st.session_state.uploaded_paths.items():
            st.write(f"- {name} → {p}")

        task = st.text_area("Task description", height=120, placeholder="E.g. Join orders.csv with customers.csv on customer_id and summarize totals")
        run_btn = st.button("Plan & Run")

    with right_col:
        st.header("Workflow")
        st.markdown(f"**Logs:** `logs/agent.log`")
        if "agent_result" not in st.session_state:
            st.session_state.agent_result = None

        if st.session_state.agent_result:
            res = st.session_state.agent_result
            plan_steps = res.get("plan") or []
            if plan_steps:
                st.subheader("Plan")
                for idx, plan_step in enumerate(plan_steps, start=1):
                    desc = plan_step.get("description") or plan_step.get("tool") or ""
                    params = plan_step.get("params") or {}
                    st.markdown(f"{idx}. **{plan_step.get('tool', 'unknown')}** — {desc}")
                    if params:
                        with st.expander(f"Parameters for step {idx}", expanded=False):
                            st.json(params)

            st.subheader("Run Output")
            results = res.get("results", []) or []
            if results:
                for idx, entry in enumerate(results, start=1):
                    step = entry.get("step") or {}
                    result = entry.get("result")
                    raw_output = entry.get("raw")
                    render_node(step, result, idx, raw_output=raw_output)
            else:
                st.info("No workflow nodes were produced. Showing raw agent result for debugging:")
                st.write(res)

            # session dfs
            if res.get("session"):
                st.subheader("Session dataframes")
                for df_id, df in res["session"].items():
                    st.markdown(f"**{df_id}** — shape: {df.shape}")
                    st.dataframe(df.head(10))
            else:
                st.info("No session dataframes were produced.")

            history_entries = res.get("history") or []
            st.subheader("Workflow Thread")
            if history_entries:
                for idx, entry in enumerate(history_entries, start=1):
                    if isinstance(entry, dict):
                        content = entry.get("content") or entry
                    else:
                        content = str(entry)
                    st.markdown(f"**{idx}.** {content}")
            else:
                st.info("No workflow history entries available for this run.")

    if run_btn:
        if not task:
            st.error("Please provide a task description.")
        else:
            with st.spinner("Running agent — this may take a while"):
                # build llm client based on user selection
                provider_label = st.session_state.get("llm_provider", provider_options[0])
                if provider_label.startswith("Ollama"):
                    llm_client = simple_llm_client_from_ollama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=settings.default_temperature)
                else:
                    if not settings.gemini_api_key:
                        st.error("Gemini API key is not configured. Please set GEMINI_API_KEY or choose the Ollama provider.")
                        return
                    llm_client = simple_llm_client_from_gemini(api_key=settings.gemini_api_key, model=settings.gemini_model, temperature=settings.default_temperature)
                agent = Agent(llm_manager=LLMManager(base_prompt_path=settings.base_prompt_path))

                # prepare uploaded_files mapping: name -> path (pandas can read path)
                uploaded_files: Dict[str, Any] = {name: Path(p) for name, p in st.session_state.uploaded_paths.items()}

                try:
                    result = agent.execute(task, llm_client, uploaded_files=uploaded_files)
                    st.session_state.agent_result = result
                    st.success("Workflow completed")
                except Exception as e:
                    err_msg = str(e)
                    if provider_label.startswith("Ollama") and "Ollama request failed" in err_msg:
                        st.error(
                            "Could not reach the Ollama server. Please start Ollama (e.g., `ollama serve` or `ollama run <model>`)."
                        )
                        st.info("Tip: switch the LLM provider to Gemini if you have an API key configured.")
                    else:
                        st.error(f"Agent execution failed: {err_msg}")
                finally:
                    # cleanup uploads per your request
                    cleanup_uploads()
                    st.session_state.uploaded_paths = {}


if __name__ == "__main__":
    main()
