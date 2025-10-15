from __future__ import annotations

import json
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from basic_agent.agent import AgentState, InteractiveAgent
from basic_agent.llm_client import LLMClient, LLMClientError
from basic_agent.tools import TOOL_REGISTRY
from basic_agent.types import ToolSuggestion

REJECTION_PROMPT = "The previous suggestion was rejected. Provide an alternate subtask or mark the task complete."
RECOVERY_PROMPT = "Tool call failed. Offer a recovery step or alternative approach."

st.set_page_config(page_title="Basic Agent UI", page_icon="🤖", layout="wide")
load_dotenv(override=False)

DEFAULT_SESSION_VALUES = {
    "api_key": "",
    "model_name": "",
    "auto_confirm": False,
    "agent_state": None,
    "llm_client": None,
    "task": "",
    "current_suggestion": None,
    "csv_path_value": "",
    "last_result": None,
    "workflow_complete": False,
    "error": None,
    "info": None,
}

for key, value in DEFAULT_SESSION_VALUES.items():
    st.session_state.setdefault(key, value)


def apply_api_key_from_state() -> None:
    api_key_raw = st.session_state.get("api_key")
    if isinstance(api_key_raw, str):
        api_key = api_key_raw.strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key


def reset_workflow_state() -> None:
    st.session_state.agent_state = None
    st.session_state.llm_client = None
    st.session_state.task = ""
    st.session_state.current_suggestion = None
    st.session_state.csv_path_value = ""
    st.session_state.last_result = None
    st.session_state.workflow_complete = False
    st.session_state.error = None
    st.session_state.info = "Workflow reset."


def start_workflow(task: str) -> None:
    task = task.strip()
    if not task:
        st.session_state.error = "Task description is required."
        return

    apply_api_key_from_state()
    if not os.getenv("GEMINI_API_KEY"):
        st.session_state.error = "Gemini API key is required. Add it in the sidebar."
        return

    model = st.session_state.get("model_name") or None
    try:
        llm = LLMClient(model=model)
    except Exception as exc:  # broad to surface config issues
        st.session_state.error = f"Failed to initialize LLM client: {exc}"
        return

    state = AgentState()
    state.push("user", f"Task: {task}")

    st.session_state.agent_state = state
    st.session_state.llm_client = llm
    st.session_state.task = task
    st.session_state.workflow_complete = False
    st.session_state.last_result = None
    st.session_state.current_suggestion = None
    st.session_state.csv_path_value = ""
    st.session_state.error = None
    st.session_state.info = "Workflow started."

    request_suggestion()


def request_suggestion() -> None:
    state = st.session_state.agent_state
    llm = st.session_state.llm_client

    if state is None or llm is None:
        st.session_state.error = "Start a workflow before requesting a suggestion."
        return
    if st.session_state.workflow_complete:
        st.session_state.info = "Workflow already marked complete."
        return

    try:
        suggestion = llm.suggestion(state.history)
    except LLMClientError as exc:
        st.session_state.error = f"LLM error: {exc}"
        return

    st.session_state.current_suggestion = suggestion
    st.session_state.csv_path_value = suggestion.input.get("path", "")
    st.session_state.error = None
    st.session_state.info = "New suggestion received."

    if st.session_state.get("auto_confirm"):
        confirm_suggestion(auto_trigger=True)


def confirm_suggestion(auto_trigger: bool = False) -> None:
    suggestion: Optional[ToolSuggestion] = st.session_state.current_suggestion
    state: Optional[AgentState] = st.session_state.agent_state

    if suggestion is None or state is None:
        return

    path_override: Optional[str] = None
    if suggestion.tool == "CSV_TO_VARIABLE":
        candidate = (st.session_state.get("csv_path_value") or "").strip()
        path_override = candidate or suggestion.input.get("path")

    execute_current_suggestion(path_override, auto_trigger=auto_trigger)


def reject_suggestion() -> None:
    suggestion = st.session_state.current_suggestion
    state = st.session_state.agent_state

    if suggestion is None or state is None:
        return

    state.push("user", REJECTION_PROMPT)
    st.session_state.current_suggestion = None
    st.session_state.info = "Suggestion rejected. Request a new one when ready."
    st.session_state.error = None


def execute_current_suggestion(path_override: Optional[str], auto_trigger: bool = False) -> None:
    suggestion = st.session_state.current_suggestion
    state = st.session_state.agent_state

    if suggestion is None or state is None:
        return

    try:
        result_text = execute_tool_action(state, suggestion, path_override)
    except Exception as exc:
        error_message = f"Tool execution failed: {exc}"
        state.push("assistant", json.dumps({"tool": suggestion.tool, "input": suggestion.input}))
        state.push("tool", error_message)
        state.push("user", RECOVERY_PROMPT)
        st.session_state.error = error_message
        st.session_state.info = None
        st.session_state.current_suggestion = None
        return

    state.push("assistant", json.dumps({"tool": suggestion.tool, "input": suggestion.input}))
    state.push("tool", result_text)

    st.session_state.last_result = result_text
    st.session_state.error = None
    st.session_state.current_suggestion = None

    if InteractiveAgent._task_is_complete(result_text, suggestion):
        st.session_state.workflow_complete = True
        st.session_state.info = "Workflow marked complete."
    else:
        st.session_state.info = "Step executed successfully." if not auto_trigger else "Step auto-confirmed and executed."


def execute_tool_action(state: AgentState, suggestion: ToolSuggestion, path_override: Optional[str]) -> str:
    tool_fn = TOOL_REGISTRY.get(suggestion.tool)
    if tool_fn is None:
        raise ValueError(f"Unknown tool: {suggestion.tool}")

    if suggestion.tool == "CSV_TO_VARIABLE":
        name = suggestion.input.get("name") or suggestion.input.get("as")
        if not isinstance(name, str) or not name:
            raise ValueError("CSV_TO_VARIABLE requires a 'name' parameter to store the DataFrame")

        path = path_override or suggestion.input.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("CSV_TO_VARIABLE requires a 'path' parameter")

        suggestion.input["path"] = path
        df = tool_fn(path)
        state.set_dataframe(name, df)
        return f"DataFrame '{name}' loaded from {path} with shape {df.shape}."

    if suggestion.tool == "SUMMARIZE_DATA":
        df_name = resolve_single_name(suggestion.input)
        df = state.get_dataframe(df_name)
        summary = tool_fn(df)
        return f"Summary for '{df_name}':\n{summary}"

    if suggestion.tool == "COMBINE_SUMMARIZE":
        names = resolve_name_list(suggestion.input)
        dfs = [state.get_dataframe(name) for name in names]
        summary = tool_fn(dfs)
        return "Combined summary:\n" + summary

    raise ValueError(f"Tool not yet implemented: {suggestion.tool}")


def resolve_single_name(payload: dict) -> str:
    for key in ("df", "name", "target", "variable"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Expected a dataframe identifier in payload")


def resolve_name_list(payload: dict) -> list[str]:
    for key in ("dfs", "inputs", "variables", "names"):
        value = payload.get(key)
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return value
    raise ValueError("Expected a non-empty list of dataframe identifiers")


def render_history(state: AgentState) -> None:
    if not state.history:
        return

    st.subheader("Conversation")
    for message in state.history:
        role = message.role
        display_role = "user" if role == "user" else "assistant"
        avatar = None
        if role == "assistant":
            avatar = "🤖"
        if role == "tool":
            display_role = "assistant"
            avatar = "🛠️"

        with st.chat_message(display_role, avatar=avatar):
            render_message_content(message)


def render_message_content(message) -> None:
    content = message.content

    if message.role == "assistant":
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if parsed is not None:
            st.json(parsed)
            return

    if message.role == "tool":
        st.markdown(content)
        return

    st.markdown(content)


def render_dataframes(state: AgentState) -> None:
    if not state.variables:
        return

    st.subheader("Stored DataFrames")
    for name, df in state.variables.items():
        with st.expander(f"{name} — shape {df.shape}"):
            st.dataframe(df.head(20))


with st.sidebar:
    st.header("Configuration")
    st.text_input("Gemini API Key", type="password", key="api_key")
    st.text_input("Model Override (optional)", key="model_name")
    st.checkbox("Auto-confirm steps", key="auto_confirm")
    st.button("Reset Workflow", on_click=reset_workflow_state, type="secondary", use_container_width=True,
              disabled=st.session_state.agent_state is None)

apply_api_key_from_state()

st.title("Basic Agent Streamlit UI")
st.write(
    "Plan and execute data-analysis workflows with Gemini. Provide a high-level task, review each suggested tool call, "
    "and run it directly from this interface."
)

if st.session_state.error:
    st.error(st.session_state.error)
if st.session_state.info:
    st.info(st.session_state.info)

state = st.session_state.agent_state

if state is None:
    with st.form("task_form"):
        task_input = st.text_area(
            "Describe your task",
            value=st.session_state.get("task", ""),
            height=140,
            placeholder="e.g., Analyze CSV files in the data/ directory",
        )
        submitted = st.form_submit_button("Start Workflow", use_container_width=True)
    if submitted:
        start_workflow(task_input)
else:
    st.markdown(f"**Current task:** {st.session_state.task}")

    if st.session_state.workflow_complete:
        st.success("Workflow complete. Reset or request a new suggestion to continue.")

    suggestion = st.session_state.current_suggestion
    controls_container = st.container()

    with controls_container:
        if suggestion is None and not st.session_state.workflow_complete:
            st.button(
                "Request Next Suggestion",
                on_click=request_suggestion,
                use_container_width=True,
            )
        elif suggestion is not None:
            st.subheader("Suggested Step")
            st.markdown(f"**Tool:** `{suggestion.tool}`")
            st.markdown(suggestion.description)
            st.code(json.dumps(suggestion.input, indent=2), language="json")

            if suggestion.tool == "CSV_TO_VARIABLE":
                st.text_input(
                    "CSV path",
                    key="csv_path_value",
                    help="Provide a path relative to the project root or an absolute path accessible to the app.",
                )

            col1, col2 = st.columns(2)
            col1.button("Run Step", on_click=confirm_suggestion, use_container_width=True)
            col2.button("Reject", on_click=reject_suggestion, type="secondary", use_container_width=True)

    if st.session_state.last_result:
        st.subheader("Latest Tool Output")
        st.code(st.session_state.last_result)

    render_history(state)
    render_dataframes(state)
