from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from basic_agent.llm_client import LLMClient, LLMClientError
from basic_agent.tools import combine_summarize, summarize_data

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
    "uploaded_signature": (),
    "loaded_frames": [],
    "join_plan": None,
    "combined_frame": None,
    "combined_summary": "",
    "final_insights": [],
    "task_plan": [],
    "plan_notes": "",
    "workflow_nodes": [],
    "auto_run_lock": False,
    "auto_run_blocked_step": None,
    "step_retry_counts": {},
    "error_message": None,
    "info_message": None,
    "llm_client": None,
    "workflow_complete": False,
}

SUPPORTED_TASK_MESSAGE = (
    "This workflow can only help with CSV analysis tasks using these actions: select_files, show_data, summarize_data, combine_data. Please submit a compatible request."
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

def request_task_plan(task: str, available_files: List[str], llm: LLMClient) -> dict:
    tools_context = "\n".join(
        [
            "Tools you can plan with:",
            "- select_files: prompt the analyst to choose/upload CSV files and load them into pandas DataFrames.",
            "- show_data: display a preview of specific loaded DataFrames (head of 20-50 rows).",
            "- summarize_data: generate per-file summaries using pandas describe-type statistics or helper functions.",
            "- combine_data: merge multiple DataFrames using join keys and produce a combined table.",
        ]
    )

    available_listing = ", ".join(sorted(available_files)) if available_files else "(no CSVs detected yet)"

    prompt = f"""
You are an analytics planning assistant. The user provided this instruction:
"{task}"

Plan the workflow using only the available tools listed below. Create actionable steps that the analyst can execute one at a time. Prefer concise modular steps that directly accomplish the user's request. If the user only asked for a single action (for example, just showing a file), return exactly that step.

{tools_context}

CSV files currently detected in the workspace: {available_listing}

Return JSON with this shape:
{{
    "steps": [
        {{
            "id": "step_1",
            "action": "select_files|show_data|summarize_data|combine_data",
            "description": "<what to do>",
            "targets": ["optional list of file names or dataframe names"],
            "rows": 20
        }}
    ],
    "notes": "<optional guidance for the analyst>"
}}

Guidelines:
- Use at most 1-3 steps unless the user explicitly asked for a longer analysis.
- Ensure step IDs are unique (step_1, step_2, ...).
- Include a `select_files` step before any action that requires CSV data unless the user clearly stated the data is already loaded.
- Only include the "rows" field when the action is "show_data" and you want to control preview length.
- For direct commands (e.g., "show data from customers.csv"), respond with a single show_data step.
- Only reference CSV files that exist or that the user clearly mentioned.
- If the task demands additional files not listed, include them in the step description but still use the select_files action.
- If no action is required, return an empty steps array.
- The JSON must be valid and include only the keys described above.
""".strip()

    return llm.json_completion(prompt)


def request_join_plan(task: str, metadata: List[dict], llm: LLMClient) -> dict:
    prompt = f"""
You are designing how to combine multiple pandas DataFrames created from CSV files for the task: "{task}".
You will receive metadata describing each DataFrame (name, shape, columns).
Decide which tables should be joined (only when the relationship makes sense) and provide additional insights the analyst should review.

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
            "comment": "<brief rationale>",
            "columns_to_keep": ["optional", "columns", "to", "retain"]
    }}
  ],
    "final_columns": ["optional", "final", "column", "list"],
  "insights": ["<insight sentence>", "..."]
}}
Guidelines:
- Use only columns that exist in the metadata.
- Only include join steps when the tables share clearly related keys or columns; otherwise leave them separate.
- If no joins are recommended, set "join_steps" to [] and choose the most informative dataframe for "start_with".
- When joins or combinations would result in a very wide table, trim to the most relevant columns using "columns_to_keep" or "final_columns".
- Apply the steps sequentially: start from "start_with" and merge the current result with each "right" dataframe in the order listed.
- Provide 1-3 insight sentences tailored to the task and available data.
- Ensure the JSON is valid.
""".strip()
    return llm.json_completion(prompt)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


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

def build_frames_from_uploads(files: Iterable) -> tuple[List[LoadedFrame], tuple]:
    used_names: set[str] = set()
    frames: List[LoadedFrame] = []
    signature_items: List[tuple[str, int]] = []

    for uploaded_file in files:
        display_name = Path(uploaded_file.name).name
        synthetic_path = f"uploaded://{display_name}"

        data = uploaded_file.getvalue()
        signature_items.append((display_name, len(data)))
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        df = pd.read_csv(io.BytesIO(data))
        df.attrs["source_path"] = synthetic_path
        name = slugify_name(display_name, used_names)
        summary = summarize_data(df)
        metadata = extract_metadata(df, name, display_name, synthetic_path)
        frames.append(
            LoadedFrame(
                name=name,
                display_name=display_name,
                path=synthetic_path,
                dataframe=df,
                summary=summary,
                metadata=metadata,
            )
        )

    signature = tuple(sorted(signature_items))
    return frames, signature


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
        retained_columns: List[str] = []
        columns_to_keep = step.get("columns_to_keep")
        if isinstance(columns_to_keep, list) and columns_to_keep:
            retained_columns = [col for col in columns_to_keep if col in current.columns]
            if retained_columns:
                current = current[retained_columns]
        applied_steps.append(
            {
                "right": right,
                "on": on,
                "how": how,
                "result_name": result_name,
                "comment": step.get("comment", ""),
                "columns_retained": retained_columns,
            }
        )
    final_columns = plan.get("final_columns")
    if isinstance(final_columns, list) and final_columns:
        desired = [col for col in final_columns if col in current.columns]
        if desired:
            current = current[desired]
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
# Planning and execution helpers
# ---------------------------------------------------------------------------

EXECUTABLE_ACTIONS = {"show_data", "summarize_data", "combine_data"}
ALLOWED_PLAN_ACTIONS = {"select_files"} | EXECUTABLE_ACTIONS
MAX_STEP_ATTEMPTS = 4


def get_plan_steps() -> List[dict]:
    return st.session_state.get("task_plan", []) or []


def find_step(step_id: str) -> Optional[dict]:
    for step in get_plan_steps():
        if step.get("id") == step_id:
            return step
    return None


def has_pending_action(action: str) -> bool:
    for step in get_plan_steps():
        if step.get("action") == action and step.get("status") != "completed":
            return True
    return False


def append_workflow_node(node: dict) -> None:
    nodes = st.session_state.get("workflow_nodes", []) or []
    nodes.append(node)
    st.session_state.workflow_nodes = nodes


def mark_step_completed(step_id: str, result: Optional[dict] = None) -> None:
    step = find_step(step_id)
    if not step:
        payload = result.copy() if isinstance(result, dict) else {}
        append_workflow_node(
            {
                "type": "step",
                "step_id": step_id,
                "description": step_id,
                "status": "completed",
                **payload,
            }
        )
        st.session_state.auto_run_blocked_step = None
        if "step_retry_counts" in st.session_state:
            st.session_state.step_retry_counts.pop(step_id, None)
        return

    step["status"] = "completed"
    if result is not None:
        step["result"] = result

    log_entry = {
        "step_id": step_id,
        "action": step.get("action"),
        "description": step.get("description", step_id),
    }
    if result:
        log_entry.update(result)

    append_workflow_node({"type": "step", "status": "completed", **log_entry})
    st.session_state.auto_run_blocked_step = None
    if "step_retry_counts" in st.session_state:
        st.session_state.step_retry_counts.pop(step_id, None)
    st.session_state.info_message = f"Completed step: {step.get('description', step_id)}"
    st.session_state.error_message = None


def mark_first_pending_select_step(frames: List[LoadedFrame]) -> bool:
    available_tokens: set[str] = set()
    for frame in frames:
        display_lower = frame.display_name.lower()
        name_lower = frame.name.lower()
        available_tokens.update({display_lower, name_lower, frame.path.lower()})
        try:
            available_tokens.add(Path(display_lower).stem.lower())
        except Exception:
            pass

    for step in get_plan_steps():
        if step.get("action") == "select_files" and step.get("status") != "completed":
            targets = step.get("targets") or []
            missing_targets: List[str] = []
            for target in targets:
                normalized = str(target).strip().lower()
                target_tokens = {normalized}
                try:
                    target_tokens.add(Path(normalized).stem.lower())
                except Exception:
                    pass
                if not target_tokens & available_tokens:
                    missing_targets.append(str(target))

            if missing_targets:
                st.session_state.auto_run_blocked_step = step.get("id")
                missing_list = ", ".join(missing_targets)
                st.session_state.info_message = f"Upload the requested CSVs via ➕: {missing_list}."
                return False

            frames_payload = []
            for frame in frames:
                frames_payload.append(
                    {
                        "display_name": frame.display_name,
                        "shape": frame.dataframe.shape,
                        "path": frame.path,
                        "dataframe_name": frame.name,
                    }
                )
            mark_step_completed(
                step.get("id", "select_files"),
                {
                    "message": f"Loaded {len(frames_payload)} dataframe(s).",
                    "frames": frames_payload,
                },
            )
            return True
    return False


def ingest_uploaded_files(uploaded_files: Iterable) -> None:
    files = list(uploaded_files or [])
    if not files:
        return

    frames, signature = build_frames_from_uploads(files)
    if not frames:
        return

    if signature == st.session_state.get("uploaded_signature"):
        return

    st.session_state.uploaded_signature = signature
    st.session_state.loaded_frames = frames
    st.session_state.workflow_complete = False
    st.session_state.auto_run_blocked_step = None
    st.session_state.error_message = None

    completed = mark_first_pending_select_step(frames)
    if not completed:
        frames_payload = []
        for frame in frames:
            frames_payload.append(
                {
                    "display_name": frame.display_name,
                    "shape": frame.dataframe.shape,
                    "path": frame.path,
                    "dataframe_name": frame.name,
                }
            )
        append_workflow_node(
            {
                "type": "load",
                "description": "Files uploaded",
                "message": f"Loaded {len(frames_payload)} dataframe(s).",
                "frames": frames_payload,
            }
        )

    if completed or get_pending_select_step() is None:
        st.session_state.info_message = f"Uploaded {len(frames)} CSV file(s)."


def resolve_frames(targets: Optional[List[str]]) -> List[LoadedFrame]:
    frames: List[LoadedFrame] = st.session_state.get("loaded_frames", []) or []
    if not targets:
        return frames

    resolved: List[LoadedFrame] = []
    normalized_targets = [str(target).strip().lower() for target in targets]
    for target in normalized_targets:
        for frame in frames:
            compare_values = {
                frame.name.lower(),
                frame.display_name.lower(),
                Path(frame.path).name.lower(),
            }
            if target in compare_values:
                resolved.append(frame)
                break
    return resolved


def get_pending_select_step() -> Optional[dict]:
    for step in get_plan_steps():
        if step.get("action") == "select_files" and step.get("status") != "completed":
            return step
    return None


def auto_run_next_step() -> bool:
    if st.session_state.get("auto_run_lock"):
        return False

    steps = get_plan_steps()
    next_step = next((step for step in steps if step.get("status") != "completed"), None)
    if not next_step:
        return False

    step_id = next_step.get("id")
    action = next_step.get("action")
    blocked = st.session_state.get("auto_run_blocked_step")
    retry_counts = st.session_state.get("step_retry_counts", {})
    current_retries = int(retry_counts.get(step_id, 0))

    if blocked and blocked == step_id and action != "select_files":
        if current_retries < MAX_STEP_ATTEMPTS:
            st.session_state.auto_run_blocked_step = None
        else:
            return False

    if action == "select_files":
        frames = st.session_state.get("loaded_frames", []) or []
        if frames:
            if mark_first_pending_select_step(frames):
                return True
        else:
            st.session_state.info_message = "Upload CSV files with the ➕ button to continue."
        st.session_state.auto_run_blocked_step = step_id
        return False

    if action in EXECUTABLE_ACTIONS:
        st.session_state.auto_run_lock = True
        try:
            execute_step(step_id)
        finally:
            st.session_state.auto_run_lock = False

        step = find_step(step_id)
        if step and step.get("status") == "completed":
            st.session_state.step_retry_counts.pop(step_id, None)
            return True

        failures = current_retries + 1
        st.session_state.step_retry_counts[step_id] = failures

        if failures < MAX_STEP_ATTEMPTS:
            st.session_state.info_message = (
                f"Retrying '{next_step.get('description', step_id)}' (attempt {failures + 1}/{MAX_STEP_ATTEMPTS})."
            )
            st.session_state.auto_run_blocked_step = None
            return True

        st.session_state.info_message = None
        st.session_state.error_message = (
            f"Unable to complete '{next_step.get('description', step_id)}' after {MAX_STEP_ATTEMPTS} attempts."
        )
        st.session_state.auto_run_blocked_step = step_id
        append_workflow_node(
            {
                "type": "abort",
                "description": "Workflow stopped",
                "message": st.session_state.error_message,
                "step_id": step_id,
                "attempts": failures,
            }
        )
        return False

    return False


def execute_step(step_id: str) -> None:
    step = find_step(step_id)
    if not step:
        st.session_state.error_message = f"Unable to find step '{step_id}'."
        return

    if step.get("status") == "completed":
        st.session_state.info_message = f"Step '{step.get('description', step_id)}' already completed."
        return

    action = step.get("action")
    try:
        if action == "show_data":
            targets = step.get("targets") or []
            frames = resolve_frames(targets)
            if not frames:
                raise ValueError("No matching dataframes found. Load the requested files first.")
            rows = step.get("rows") or 20
            frame_payload = []
            for frame in frames:
                preview = frame.dataframe.head(int(rows))
                frame_payload.append(
                    {
                        "display_name": frame.display_name,
                        "shape": frame.dataframe.shape,
                        "preview": preview,
                        "path": frame.path,
                        "dataframe_name": frame.name,
                    }
                )
            mark_step_completed(step_id, {"frames": frame_payload, "message": "Displayed data preview."})

        elif action == "summarize_data":
            targets = step.get("targets") or []
            frames = resolve_frames(targets)
            if not frames:
                raise ValueError("No matching dataframes found to summarize. Load the requested files first.")
            frame_payload = []
            for frame in frames:
                frame_payload.append(
                    {
                        "display_name": frame.display_name,
                        "shape": frame.dataframe.shape,
                        "summary": frame.summary,
                        "path": frame.path,
                        "dataframe_name": frame.name,
                    }
                )
            mark_step_completed(step_id, {"frames": frame_payload, "message": "Summaries ready."})

        elif action == "combine_data":
            apply_api_key_from_state()
            llm = get_llm_client()
            if llm is None:
                return

            targets = step.get("targets") or []
            frames = resolve_frames(targets)
            if not frames:
                frames = st.session_state.get("loaded_frames", []) or []
            if not frames:
                raise ValueError("Load the dataframes you want to combine before running this step.")

            frame_map = {frame.name: frame for frame in frames}
            metadata = [frame.metadata for frame in frames]
            plan = request_join_plan(st.session_state.task, metadata, llm)
            combined_df, applied_steps = combine_via_plan(frames, plan)

            if not plan.get("join_steps"):
                summary = combine_summarize([frame.dataframe for frame in frames])
            else:
                summary = summarize_data(combined_df)

            ordered_names: List[str] = []
            start_with = plan.get("start_with")
            if isinstance(start_with, str) and start_with in frame_map:
                ordered_names.append(start_with)
            for join_step in plan.get("join_steps", []):
                right = join_step.get("right")
                if isinstance(right, str) and right in frame_map and right not in ordered_names:
                    ordered_names.append(right)
            if not ordered_names:
                ordered_names = [frame.name for frame in frames]
            joined_tables = [frame_map[name].display_name for name in ordered_names if name in frame_map]

            st.session_state.join_plan = plan | {"applied_steps": applied_steps}
            st.session_state.combined_frame = combined_df
            st.session_state.combined_summary = summary
            st.session_state.workflow_complete = True

            metadata_warning = None
            try:
                write_relations_metadata(
                    st.session_state.task,
                    frames,
                    st.session_state.join_plan,
                    st.session_state.get("final_insights", []),
                    summary,
                )
            except Exception as metadata_exc:  # pragma: no cover - IO errors
                metadata_warning = f"Combined data ready, but failed to update metadata: {metadata_exc}"

            message = "Combined dataset created."
            if metadata_warning:
                message += " (See warning below.)"

            mark_step_completed(
                step_id,
                {
                    "message": message,
                    "combined_preview": combined_df.head(50),
                    "summary": summary,
                    "join_steps": applied_steps,
                    "joined_tables": joined_tables,
                },
            )

            if metadata_warning:
                st.session_state.error_message = metadata_warning

        else:
            mark_step_completed(step_id, {"message": f"No automated handler for action '{action}'."})

    except Exception as exc:  # pragma: no cover - runtime execution paths
        error_text = f"Failed to execute step '{step.get('description', step_id)}': {exc}"
        st.session_state.error_message = error_text
        st.session_state.auto_run_blocked_step = step_id
        append_workflow_node(
            {
                "type": "error",
                "description": step.get("description", step_id),
                "step_id": step_id,
                "action": action,
                "status": "error",
                "message": error_text,
            }
        )

def render_workflow_nodes() -> None:
    st.subheader("Workflow thread")

    pending_select = get_pending_select_step()
    if pending_select and not st.session_state.get("loaded_frames"):
        st.info("Upload CSV files with the ➕ button to satisfy the plan's file step.")

    nodes = st.session_state.get("workflow_nodes", []) or []
    if not nodes:
        st.caption("No workflow activity yet.")
        return

    for node in reversed(nodes):
        node_type = node.get("type", "step")
        header = node.get("description") or node.get("step_id") or node_type.title()
        with st.chat_message("assistant"):
            st.markdown(f"**{header}**")

            if node_type == "prompt":
                prompt_text = node.get("message") or ""
                if prompt_text:
                    st.write(prompt_text)
                continue

            if node_type == "step":
                action = node.get("action")
                status = node.get("status", "completed")
                meta_chunks = []
                if action:
                    meta_chunks.append(f"Action: `{action}`")
                if status:
                    meta_chunks.append(f"Status: {status}")
                if meta_chunks:
                    st.caption(" · ".join(meta_chunks))
            elif node_type == "plan":
                plan_steps = node.get("steps") or []
                if plan_steps:
                    st.caption("Planned steps:")
                    for plan_step in plan_steps:
                        action = plan_step.get("action")
                        description = plan_step.get("description") or plan_step.get("id")
                        st.markdown(f"- `{action}` · {description}")

            message = node.get("message")
            if message:
                st.write(message)

            frames_info = node.get("frames") or []
            for frame_info in frames_info:
                title = frame_info.get("display_name") or "Dataframe"
                shape = frame_info.get("shape")
                dataframe_name = frame_info.get("dataframe_name")
                path = frame_info.get("path")
                meta_parts = []
                if shape:
                    meta_parts.append(f"shape {shape}")
                if dataframe_name:
                    meta_parts.append(f"id `{dataframe_name}`")
                if path:
                    meta_parts.append(f"source `{path}`")
                if meta_parts:
                    st.caption(f"{title} — " + " · ".join(meta_parts))
                else:
                    st.caption(title)
                summary_text = frame_info.get("summary")
                if summary_text:
                    st.code(summary_text)
                preview = frame_info.get("preview")
                if preview is not None:
                    st.dataframe(preview)

            combined_preview = node.get("combined_preview")
            if combined_preview is not None:
                st.caption("Combined dataset preview")
                st.dataframe(combined_preview)

            summary_text = node.get("summary")
            if summary_text:
                st.caption("Summary")
                st.code(summary_text)

            join_steps = node.get("join_steps") or []
            if join_steps:
                st.caption("Join steps")
                for step_info in join_steps:
                    details = ", ".join(step_info.get("on", []))
                    retained_cols = step_info.get("columns_retained") or []
                    columns_note = ""
                    if retained_cols:
                        columns_note = f" Columns kept: {', '.join(retained_cols)}."
                    st.markdown(
                        f"- Joined `{step_info.get('right')}` on [{details}] using **{step_info.get('how')}** → `{step_info.get('result_name')}`. {step_info.get('comment', '')}{columns_note}"
                    )

            joined_tables = node.get("joined_tables") or []
            if joined_tables:
                st.caption("Joined tables")
                st.write(", ".join(joined_tables))

            insights = node.get("insights") or []
            if insights:
                st.caption("Insights")
                for insight in insights:
                    st.markdown(f"- {insight}")


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


def render_loaded_summaries(frames: List[LoadedFrame]) -> None:
    st.subheader("Loaded dataframes")
    for idx, frame in enumerate(frames, start=1):
        st.markdown(f"**{idx}. {frame.display_name}** (`{frame.name}`) — shape {frame.dataframe.shape}")
        st.caption(f"Source path: `{frame.path}`")
        st.code(frame.summary)
        st.dataframe(frame.dataframe.head(20))


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
        available_files = [
            f"{frame.display_name} (dataframe id: {frame.name})"
            for frame in st.session_state.get("loaded_frames", [])
        ]
        plan = request_task_plan(task, available_files, llm)
    except LLMClientError as exc:
        st.session_state.error_message = str(exc)
        return

    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    parsed_steps: List[dict] = []
    invalid_actions: List[str] = []
    for idx, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            invalid_actions.append("<invalid step>")
            continue

        action = str(raw_step.get("action") or "").strip().lower()
        if not action or action not in ALLOWED_PLAN_ACTIONS:
            invalid_actions.append(action or "<missing action>")
            continue

        step_id = str(raw_step.get("id") or f"step_{idx + 1}")
        step_copy = dict(raw_step)
        step_copy["id"] = step_id
        step_copy["action"] = action
        step_copy["status"] = "pending"
        parsed_steps.append(step_copy)

    if invalid_actions or not parsed_steps:
        allowed_actions_text = ", ".join(sorted(ALLOWED_PLAN_ACTIONS))
        if invalid_actions:
            unsupported_list = ", ".join(sorted({item for item in invalid_actions}))
            message = (
                "Task rejected. The planner attempted to use unsupported actions: "
                f"{unsupported_list}. Allowed actions: {allowed_actions_text}."
            )
        else:
            message = (
                "Task rejected. No compatible CSV actions could be planned. "
                f"Allowed actions: {allowed_actions_text}."
            )

        st.session_state.task = task
        st.session_state.error_message = message
        st.session_state.info_message = None
        st.session_state.task_plan = []
        st.session_state.plan_notes = ""
        st.session_state.workflow_nodes = []
        st.session_state.workflow_complete = False
        st.session_state.auto_run_blocked_step = None
        st.session_state.auto_run_lock = False
        st.session_state.join_plan = None
        st.session_state.combined_frame = None
        st.session_state.combined_summary = ""
        st.session_state.final_insights = []
        return

    st.session_state.task = task
    steps = parsed_steps

    st.session_state.task_plan = steps
    st.session_state.plan_notes = plan.get("notes", "")
    st.session_state.workflow_nodes = []
    st.session_state.pop("execution_log", None)
    st.session_state.join_plan = None
    st.session_state.combined_frame = None
    st.session_state.combined_summary = ""
    st.session_state.final_insights = []
    st.session_state.workflow_complete = False
    st.session_state.auto_run_blocked_step = None
    st.session_state.auto_run_lock = False
    needs_select_step = any(step.get("action") == "select_files" for step in steps)
    append_workflow_node(
        {
            "type": "prompt",
            "description": "Task prompt",
            "message": task,
        }
    )
    append_workflow_node(
        {
            "type": "plan",
            "description": "Plan generated",
            "message": f"Prepared {len(steps)} step(s) for the task.",
            "steps": [
                {
                    "id": step.get("id"),
                    "action": step.get("action"),
                    "description": step.get("description", step.get("id")),
                }
                for step in steps
            ],
            "notes": st.session_state.plan_notes,
        }
    )

    if needs_select_step:
        st.session_state.info_message = "Plan accepted. Use the ➕ uploader to provide the requested CSVs."
    else:
        st.session_state.info_message = "Plan accepted. Executing applicable steps automatically."
    st.session_state.error_message = None
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

reset_request = st.session_state.pop(RESET_TRIGGER_KEY, None)
if reset_request:
    message = reset_request.get("message") if isinstance(reset_request, dict) else "Workflow reset."
    kind = reset_request.get("kind") if isinstance(reset_request, dict) else "info"
    st.session_state.pop("prompt_uploader_widget", None)
    st.session_state.update(
        {
            "task": "",
            "loaded_frames": [],
            "uploaded_signature": (),
            "join_plan": None,
            "combined_frame": None,
            "combined_summary": "",
            "final_insights": [],
            "task_plan": [],
            "plan_notes": "",
            "workflow_nodes": [],
            "auto_run_lock": False,
            "auto_run_blocked_step": None,
            "info_message": None,
            "error_message": None,
            "workflow_complete": False,
        }
    )
    st.session_state.pop("execution_log", None)
    if kind == "error":
        st.session_state.error_message = message
        st.session_state.info_message = None
    else:
        st.session_state.info_message = message
        st.session_state.error_message = None
    st.rerun()

st.title("Basic Agent Workflow")
st.write(
    "Describe your analytics task or provide a direct command. The agent will draft a plan using the available CSV tooling, "
    "and you can execute the steps one by one."
)

render_messages()

col_prompt, col_upload = st.columns([5, 1])
with col_prompt:
    task_input_value = st.text_area(
        "Describe your task or command",
        value=st.session_state.get("task", ""),
        height=140,
        placeholder="e.g., Show data in customers.csv or combine customer and order information",
    )
with col_upload:
    uploaded_files = st.file_uploader(
        "➕ Upload CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="prompt_uploader_widget",
    )

if uploaded_files:
    ingest_uploaded_files(uploaded_files)

plan_steps = get_plan_steps()
plan_in_progress = any(step.get("status") != "completed" for step in plan_steps)

button_col, run_col = st.columns([3, 1])
with button_col:
    plan_disabled = plan_in_progress and bool(plan_steps)
    if st.button("Plan & Execute Task", use_container_width=True, disabled=plan_disabled):
        handle_prompt_submission(task_input_value)

with run_col:
    run_disabled = not plan_in_progress
    if st.button("Run Next Step", use_container_width=True, disabled=run_disabled):
        if auto_run_next_step():
            st.rerun()
        else:
            if not st.session_state.get("info_message") and not st.session_state.get("error_message"):
                st.session_state.info_message = "No executable steps available. Upload required data or reset the workflow."

if auto_run_next_step():
    st.rerun()

if st.session_state.get("task"):
    st.caption(f"Current task: **{st.session_state.task}**")

st.divider()
render_workflow_nodes()
