from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from basic_agent.llm_client import LLMClient, LLMClientError
from basic_agent.tools import combine_summarize, csv_to_variable, summarize_data

st.set_page_config(page_title="Basic Agent Workflow", page_icon="🤖", layout="wide")
load_dotenv(override=False)


@dataclass
class LoadedFrame:
    name: str
    display_name: str
    path: str
    dataframe: pd.DataFrame
    summary: str
    metadata: dict


DEFAULT_SESSION_VALUES = {
    "api_key": "",
    "model_name": "",
    "task": "",
    "workflow_step": "prompt",
    "file_plan": None,
    "available_local_files": [],
    "folder_value": "data",
    "selected_file_ids": [],
    "uploaded_file_map": {},
    "upload_temp_dir": "data/uploads",
    "last_folder": "",
    "loaded_frames": [],
    "join_plan": None,
    "combined_frame": None,
    "combined_summary": "",
    "final_insights": [],
    "error_message": None,
    "info_message": None,
    "llm_client": None,
    "workflow_complete": False,
}

SUPPORTED_TASK_MESSAGE = (
    "This workflow only supports loading CSV data into pandas DataFrames, displaying previews, summarizing each file, and generating combined summaries."
)

RESET_TRIGGER_KEY = "_workflow_reset_request"


for key, default in DEFAULT_SESSION_VALUES.items():
    st.session_state.setdefault(key, default)

st.session_state.setdefault(RESET_TRIGGER_KEY, None)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def apply_api_key_from_state() -> None:
    api_key = str(st.session_state.get("api_key") or "").strip()
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key


def reset_workflow_state(message: Optional[str] = "Workflow reset.", kind: str = "info") -> None:
    st.session_state[RESET_TRIGGER_KEY] = {"message": message, "kind": kind}
    clear_upload_dir()


def is_supported_prompt(task: str) -> bool:
    normalized = " ".join(task.lower().split())
    if not normalized:
        return False

    command_tokens = {
        "ls",
        "pwd",
        "cd",
        "mkdir",
        "rm",
        "rmdir",
        "touch",
        "git",
        "pip",
        "python",
        "sudo",
        "apt",
        "docker",
        "curl",
        "wget",
        "chmod",
    }
    first_token = normalized.split(" ", 1)[0]
    if first_token in command_tokens:
        return False

    disallowed_fragments = ["&&", "||", ";", "`", "$(", "../", "rm -", "shutdown", "reboot"]
    if any(fragment in normalized for fragment in disallowed_fragments):
        return False

    allowed_keywords = [
        "csv",
        "data",
        "dataset",
        "dataframe",
        "table",
        "tables",
        "file",
        "files",
        "summar",
        "analy",
        "insight",
        "join",
        "merge",
        "combine",
    ]
    return any(keyword in normalized for keyword in allowed_keywords)


def get_llm_client() -> Optional[LLMClient]:
    client = st.session_state.get("llm_client")
    if client is not None:
        return client

    model = st.session_state.get("model_name") or None
    try:
        client = LLMClient(model=model)
    except Exception as exc:  # pragma: no cover - configuration or API errors
        st.session_state.error_message = f"Failed to initialize LLM client: {exc}"
        return None

    st.session_state.llm_client = client
    return client


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def request_file_plan(task: str, llm: LLMClient) -> dict:
    prompt = f"""
You are assisting with an analytics workflow. The user provided the high-level task below.
Task: "{task}"

Return a JSON object with the following structure:
{{
  "file_expectations": [
    {{"description": "<what data is needed>", "suggested_name": "<friendly label>"}}
  ],
  "notes": "<short guidance for the analyst>"
}}
Only include keys shown above. Make sure the JSON is valid.
""".strip()
    return llm.json_completion(prompt)


def request_join_plan(task: str, metadata: List[dict], llm: LLMClient) -> dict:
    prompt = f"""
You are designing how to combine multiple pandas DataFrames created from CSV files for the task: "{task}".
You will receive metadata describing each DataFrame (name, shape, columns).
Decide how to join them and provide additional insights the analyst should review.

Metadata:
{json.dumps(metadata, indent=2)}

Respond with JSON in this exact shape:
{{
  "start_with": "<dataframe name to use as the initial table>",
  "join_steps": [
    {{
      "right": "<dataframe name to join>",
      "on": ["column1", "column2"],
      "how": "inner|left|right|outer",
      "result_name": "<name for the merged result after this step>",
      "comment": "<brief rationale>"
    }}
  ],
  "insights": ["<insight sentence>", "..."]
}}
Guidelines:
- Use only columns that exist in the metadata.
- If no joins are recommended, set "join_steps" to [] and choose the most informative dataframe for "start_with".
- Apply the steps sequentially: start from "start_with" and merge the current result with each "right" dataframe in the order listed.
- Provide 1-3 insight sentences tailored to the task and available data.
- Ensure the JSON is valid.
""".strip()
    return llm.json_completion(prompt)


def request_extra_insights(task: str, combined_summary: str, llm: LLMClient) -> List[str]:
    prompt = f"""
The analyst executed the task "{task}" and produced this combined dataframe summary:
---
{combined_summary}
---
Return a JSON object with a single key "insights" whose value is a list of concise follow-up insights or recommendations.
Keep each insight under 160 characters.
""".strip()
    response = llm.json_completion(prompt)
    insights = response.get("insights")
    if isinstance(insights, list) and insights:
        return [str(item) for item in insights]
    return []


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def ensure_upload_dir() -> Path:
    upload_dir = Path(st.session_state.get("upload_temp_dir", "data/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def clear_upload_dir() -> None:
    upload_dir = ensure_upload_dir()
    for entry in upload_dir.iterdir():
        if entry.is_file():
            entry.unlink(missing_ok=True)  # type: ignore[arg-type]
        elif entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def list_csv_files(folder: str) -> List[str]:
    try:
        path = Path(folder).expanduser().resolve()
    except Exception:
        return []
    if not path.exists() or not path.is_dir():
        return []
    return sorted([entry.name for entry in path.iterdir() if entry.is_file() and entry.suffix.lower() == ".csv"])


def save_uploaded_files(files: Iterable) -> Dict[str, str]:
    mapping: Dict[str, str] = st.session_state.get("uploaded_file_map", {}) or {}
    upload_dir = ensure_upload_dir()

    for uploaded_file in files:
        filename = Path(uploaded_file.name).name
        target = upload_dir / filename
        counter = 1
        while target.exists():
            target = upload_dir / f"{target.stem}_{counter}{target.suffix}"
            counter += 1

        data = uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read()
        with open(target, "wb") as fh:
            fh.write(data)

        mapping[f"upload::{target.name}"] = str(target)

    st.session_state.uploaded_file_map = mapping
    return mapping


def slugify_name(label: str, used: set[str]) -> str:
    base = "df_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in Path(label).stem)
    base = "_".join(filter(None, base.split("_")))
    if not base:
        base = "df_table"
    candidate = base
    counter = 1
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def extract_metadata(df: pd.DataFrame, name: str, display_name: str, path: str) -> dict:
    columns = []
    for col in df.columns:
        series = df[col]
        sample_values = series.dropna().astype(str).head(3).tolist()
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "sample_values": sample_values,
            }
        )

    return {
        "name": name,
        "display_name": display_name,
        "path": path,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
    }


def load_selected_frames(selection: List[str], folder: str) -> List[LoadedFrame]:
    folder_path = Path(folder).expanduser().resolve()
    uploaded_map = st.session_state.get("uploaded_file_map", {}) or {}
    used_names: set[str] = set()
    frames: List[LoadedFrame] = []

    for identifier in selection:
        if identifier.startswith("upload::"):
            display_name = identifier.split("::", 1)[1]
            resolved_path = uploaded_map.get(identifier)
        else:
            display_name = identifier
            resolved_path = str((folder_path / identifier).resolve())

        if not resolved_path:
            raise FileNotFoundError(f"Unable to resolve path for {identifier}")

        df = csv_to_variable(resolved_path)
        name = slugify_name(display_name, used_names)
        summary = summarize_data(df)
        metadata = extract_metadata(df, name, display_name, resolved_path)
        frames.append(LoadedFrame(name=name, display_name=display_name, path=resolved_path, dataframe=df, summary=summary, metadata=metadata))

    return frames
def combine_via_plan(frames: List[LoadedFrame], plan: dict) -> tuple[pd.DataFrame, List[dict]]:
    df_map = {frame.name: frame.dataframe for frame in frames}
    start_with = plan.get("start_with")
    if not isinstance(start_with, str) or start_with not in df_map:
        raise ValueError("Join plan missing or referencing unknown start dataframe")

    current = df_map[start_with].copy()
    applied_steps: List[dict] = []

    for step in plan.get("join_steps", []):
        if not isinstance(step, dict):
            continue
        right = step.get("right")
        on = step.get("on") or []
        how = str(step.get("how") or "inner").lower()
        result_name = step.get("result_name") or f"merge_{right}"

        if not isinstance(right, str) or right not in df_map:
            raise ValueError(f"Join step references unknown dataframe: {right}")
        if not on or not all(isinstance(col, str) for col in on):
            raise ValueError(f"Join step must specify column list for dataframe {right}")
        if how not in {"inner", "left", "right", "outer"}:
            raise ValueError(f"Unsupported join type '{how}'")

        right_df = df_map[right]
        current = current.merge(right_df, how=how, on=on)
        applied_steps.append(
            {
                "right": right,
                "on": on,
                "how": how,
                "result_name": result_name,
                "comment": step.get("comment", ""),
            }
        )

    return current, applied_steps


def write_relations_metadata(task: str, frames: List[LoadedFrame], join_plan: dict, insights: List[str], combined_summary: str) -> None:
    payload = {
        "task": task,
        "files": [frame.metadata for frame in frames],
        "join_plan": join_plan,
        "combined_summary": combined_summary,
        "insights": insights,
    }
    path = Path("data") / "csv_relations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_messages() -> None:
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    if st.session_state.info_message:
        st.info(st.session_state.info_message)
    if st.session_state.get("workflow_complete"):
        st.success("Workflow marked complete.")


def render_file_expectations(plan: dict) -> None:
    expectations = plan.get("file_expectations") or []
    if not expectations:
        st.warning("LLM did not provide specific file expectations. Continue by supplying relevant CSVs.")
        return

    st.subheader("LLM file expectations")
    for item in expectations:
        if not isinstance(item, dict):
            continue
        desc = item.get("description", "(no description)")
        name = item.get("suggested_name", "Unnamed file")
        st.markdown(f"- **{name}**: {desc}")

    notes = plan.get("notes")
    if isinstance(notes, str) and notes.strip():
        st.markdown(f"_Notes_: {notes.strip()}")


def render_loaded_summaries(frames: List[LoadedFrame]) -> None:
    st.subheader("Loaded dataframes")
    for idx, frame in enumerate(frames, start=1):
        st.markdown(f"**{idx}. {frame.display_name}** (`{frame.name}`) — shape {frame.dataframe.shape}")
        st.caption(f"Source path: `{frame.path}`")
        st.code(frame.summary)
        st.dataframe(frame.dataframe.head(20))


def render_combined_output(combined: pd.DataFrame, summary: str, insights: List[str], join_steps: List[dict]) -> None:
    st.subheader("Combined dataset preview")
    st.dataframe(combined.head(50))

    st.subheader("Combined summary")
    st.code(summary)

    if join_steps:
        st.subheader("Join steps executed")
        for step in join_steps:
            details = ", ".join(step.get("on", []))
            st.markdown(
                f"- Joined `{step.get('right')}` using columns [{details}] with **{step.get('how')}** join → `{step.get('result_name')}`. {step.get('comment', '')}"
            )

    if insights:
        st.subheader("Extra insights")
        for insight in insights:
            st.markdown(f"- {insight}")


# ---------------------------------------------------------------------------
# Workflow handlers
# ---------------------------------------------------------------------------

def handle_prompt_submission(task: str) -> None:
    task = task.strip()
    if not task:
        st.session_state.error_message = "Provide a task prompt before proceeding."
        return

    if not is_supported_prompt(task):
        reset_workflow_state(SUPPORTED_TASK_MESSAGE, kind="error")
        st.rerun()
        return

    st.session_state.workflow_complete = False
    apply_api_key_from_state()
    if not os.getenv("GEMINI_API_KEY"):
        st.session_state.error_message = "Gemini API key is required. Add it in the sidebar."
        return

    llm = get_llm_client()
    if llm is None:
        return

    try:
        plan = request_file_plan(task, llm)
    except LLMClientError as exc:
        st.session_state.error_message = str(exc)
        return

    st.session_state.task = task
    st.session_state.file_plan = plan
    st.session_state.workflow_step = "select"
    st.session_state.info_message = "Review the suggested files, then upload or select the actual CSVs."
    st.session_state.error_message = None
    st.rerun()


def handle_load_and_summarize(folder: str, selection: List[str]) -> None:
    if not selection:
        st.session_state.error_message = "Select at least one CSV file before loading."
        return

    st.session_state.workflow_complete = False
    try:
        frames = load_selected_frames(selection, folder)
    except Exception as exc:
        st.session_state.error_message = f"Failed to load files: {exc}"
        return

    st.session_state.loaded_frames = frames
    st.session_state.workflow_step = "summarize"
    st.session_state.info_message = "Dataframes loaded and summarized. Review the details before combining."
    st.session_state.error_message = None
    st.rerun()


def handle_generate_combined() -> None:
    frames: List[LoadedFrame] = st.session_state.get("loaded_frames", []) or []
    if not frames:
        st.session_state.error_message = "Load at least one dataframe before generating combined insights."
        return

    st.session_state.workflow_complete = False
    llm = get_llm_client()
    if llm is None:
        return

    metadata = [frame.metadata for frame in frames]

    try:
        plan = request_join_plan(st.session_state.task, metadata, llm)
    except LLMClientError as exc:
        st.session_state.error_message = str(exc)
        return

    try:
        combined_df, applied_steps = combine_via_plan(frames, plan)
    except Exception as exc:
        st.session_state.error_message = f"Failed to execute join plan: {exc}"
        return

    if not plan.get("join_steps"):
        combined_summary = combine_summarize([frame.dataframe for frame in frames])
    else:
        combined_summary = summarize_data(combined_df)

    try:
        insights = request_extra_insights(st.session_state.task, combined_summary, llm)
    except LLMClientError:
        insights = []

    st.session_state.join_plan = plan | {"applied_steps": applied_steps}
    st.session_state.combined_frame = combined_df
    st.session_state.combined_summary = combined_summary
    st.session_state.final_insights = insights
    st.session_state.workflow_step = "combined"
    st.session_state.workflow_complete = True
    st.session_state.info_message = "Combined insights generated. Workflow complete."
    st.session_state.error_message = None

    try:
        write_relations_metadata(st.session_state.task, frames, st.session_state.join_plan, insights, combined_summary)
    except Exception as exc:
        st.session_state.error_message = f"Combined results generated, but failed to update metadata file: {exc}"
    st.rerun()


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    st.text_input("Gemini API Key", type="password", key="api_key")
    st.text_input("Model Override (optional)", key="model_name")
    if st.button("Reset Workflow", use_container_width=True):
        reset_workflow_state()
        st.rerun()

st.title("Basic Agent Guided Workflow")
st.write(
    "Execute the analytics workflow step-by-step: describe your task, provide relevant CSV files, review per-file summaries, "
    "and let the LLM propose how to join everything for combined insights."
)

render_messages()

step = st.session_state.workflow_step

reset_request = st.session_state.pop(RESET_TRIGGER_KEY, None)
if reset_request:
    message = reset_request.get("message") if isinstance(reset_request, dict) else "Workflow reset."
    kind = reset_request.get("kind") if isinstance(reset_request, dict) else "info"
    st.session_state.pop("selected_folder_input", None)
    st.session_state.update(
        {
            "task": "",
            "workflow_step": "prompt",
            "file_plan": None,
            "available_local_files": [],
            "folder_value": "data",
            "selected_file_ids": [],
            "uploaded_file_map": {},
            "last_folder": "",
            "loaded_frames": [],
            "join_plan": None,
            "combined_frame": None,
            "combined_summary": "",
            "final_insights": [],
            "info_message": None,
            "error_message": None,
            "workflow_complete": False,
        }
    )
    if kind == "error":
        st.session_state.error_message = message
        st.session_state.info_message = None
    else:
        st.session_state.info_message = message
        st.session_state.error_message = None
    st.rerun()

if step == "prompt":
    with st.form("prompt_form"):
        task_input = st.text_area(
            "Describe your task",
            value=st.session_state.get("task", ""),
            height=140,
            placeholder="e.g., Analyse the files to understand customer purchasing patterns",
        )
        submitted = st.form_submit_button("Start workflow", use_container_width=True)
    if submitted:
        handle_prompt_submission(task_input)

elif step == "select":
    st.markdown(f"**Task:** {st.session_state.task}")
    render_file_expectations(st.session_state.file_plan or {})

    folder = st.text_input(
        "Folder to browse for CSVs",
        value=st.session_state.get("folder_value", "data"),
        key="selected_folder_input",
    )
    st.session_state.folder_value = folder

    if folder != st.session_state.get("last_folder"):
        st.session_state.last_folder = folder
        st.session_state.available_local_files = list_csv_files(folder)

    if st.button("Refresh local CSV list", use_container_width=True):
        st.session_state.available_local_files = list_csv_files(folder)
        st.session_state.info_message = f"Found {len(st.session_state.available_local_files)} CSV file(s) in {folder}."
        st.rerun()

    local_files = st.session_state.get("available_local_files", []) or []
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Save uploaded files", use_container_width=True):
            mapping = save_uploaded_files(uploaded_files)
            st.session_state.info_message = f"Saved {len(mapping)} uploaded file(s)."
            st.rerun()

    uploaded_options = sorted((st.session_state.get("uploaded_file_map") or {}).keys())
    options = local_files + [item for item in uploaded_options if item not in local_files]
    current_selection = [item for item in st.session_state.get("selected_file_ids", []) if item in options]
    selection = st.multiselect(
        "Select files to load",
        options=options,
        default=current_selection,
    )

    if selection != st.session_state.get("selected_file_ids"):
        st.session_state.selected_file_ids = selection

    col1, col2 = st.columns(2)
    if col1.button("Load and summarize", use_container_width=True):
        handle_load_and_summarize(folder, selection)
    if col2.button("Back to prompt", use_container_width=True):
        reset_workflow_state()
        st.rerun()

elif step == "summarize":
    frames: List[LoadedFrame] = st.session_state.get("loaded_frames", []) or []
    if not frames:
        st.warning("No dataframes loaded. Return to the previous step to select files.")
    else:
        render_loaded_summaries(frames)

    col1, col2 = st.columns(2)
    if col1.button("Generate combined insights", use_container_width=True):
        handle_generate_combined()
    if col2.button("Back to file selection", use_container_width=True):
        st.session_state.workflow_step = "select"
        st.session_state.workflow_complete = False
        st.rerun()

elif step == "combined":
    combined_df: Optional[pd.DataFrame] = st.session_state.get("combined_frame")
    if combined_df is None:
        st.warning("Combined dataframe is not available. Return to previous steps.")
    else:
        render_combined_output(
            combined_df,
            st.session_state.get("combined_summary", ""),
            st.session_state.get("final_insights", []),
            (st.session_state.get("join_plan") or {}).get("applied_steps", []),
        )

    col1, col2 = st.columns(2)
    if col1.button("Re-run combination", use_container_width=True):
        st.session_state.workflow_step = "summarize"
        st.session_state.workflow_complete = False
        st.rerun()
    if col2.button("Start over", use_container_width=True):
        reset_workflow_state()
        st.rerun()

else:
    st.warning("Unknown workflow state. Resetting.")
    reset_workflow_state()
