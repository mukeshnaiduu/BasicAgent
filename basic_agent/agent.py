from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .llm_client import ChatMessage, LLMClient, LLMClientError
from .tools import TOOL_REGISTRY
from .types import ToolSuggestion


@dataclass
class AgentState:
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[ChatMessage] = field(default_factory=list)

    def push(self, role: str, content: str) -> None:
        self.history.append(ChatMessage(role=role, content=content))

    def get_dataframe(self, name: str) -> pd.DataFrame:
        value = self.variables.get(name)
        if not isinstance(value, pd.DataFrame):
            raise KeyError(f"DataFrame '{name}' is not available")
        return value

    def set_dataframe(self, name: str, df: pd.DataFrame) -> None:
        self.variables[name] = df


class InteractiveAgent:
    """Interactive CLI agent that confirms each LLM-suggested tool with the user."""

    def __init__(self, llm_client: Optional[LLMClient] = None, auto_confirm: bool = False) -> None:
        self._llm = llm_client or LLMClient()
        self._state = AgentState()
        self._auto_confirm = auto_confirm

    def run(self, task: str, max_iterations: int = 25) -> None:
        print(f"\nStarting interactive workflow for task: {task}\n")
        self._state.push("user", f"Task: {task}")

        for iteration in range(1, max_iterations + 1):
            print(f"--- Iteration {iteration} ---")
            suggestion = self._request_suggestion()
            if suggestion is None:
                print("No further suggestions. Workflow complete.")
                return

            should_execute = self._confirm_execution(suggestion)
            if not should_execute:
                self._state.push(
                    "user",
                    "The previous suggestion was rejected. Provide an alternate subtask or mark the task complete.",
                )
                continue

            try:
                result_text = self._execute_tool(suggestion)
            except Exception as exc:
                error_message = f"Tool execution failed: {exc}"
                print(error_message)
                self._state.push("assistant", json.dumps({"tool": suggestion.tool, "input": suggestion.input}))
                self._state.push("tool", error_message)
                self._state.push(
                    "user",
                    "Tool call failed. Offer a recovery step or alternative approach.",
                )
                continue

            self._state.push("assistant", json.dumps({"tool": suggestion.tool, "input": suggestion.input}))
            self._state.push("tool", result_text)
            print(result_text)

            if self._task_is_complete(result_text, suggestion):
                print("Workflow marked complete by LLM.")
                return

        print("Reached iteration limit without completion.")

    def _request_suggestion(self) -> Optional[ToolSuggestion]:
        try:
            return self._llm.suggestion(self._state.history)
        except LLMClientError as exc:
            print(f"LLM error: {exc}")
            return None

    def _confirm_execution(self, suggestion: ToolSuggestion) -> bool:
        print("LLM Suggestion:")
        print(f"Tool: {suggestion.tool}")
        print(f"Input: {json.dumps(suggestion.input, indent=2)}")
        print(f"Description: {suggestion.description}")

        if self._auto_confirm:
            print("Auto-confirm enabled. Executing step.\n")
            return True

        while True:
            choice = input("Run this step? (y/n/exit): ").strip().lower()
            if choice in {"y", "yes"}:
                print()
                return True
            if choice in {"n", "no"}:
                print("Suggestion declined.\n")
                return False
            if choice in {"exit", "quit"}:
                raise SystemExit("Workflow aborted by user.")
            print("Please answer with 'y', 'n', or 'exit'.")

    def _execute_tool(self, suggestion: ToolSuggestion) -> str:
        tool_fn = TOOL_REGISTRY.get(suggestion.tool)
        if tool_fn is None:
            raise ValueError(f"Unknown tool: {suggestion.tool}")

        if suggestion.tool == "CSV_TO_VARIABLE":
            path = suggestion.input.get("path")
            name = suggestion.input.get("name") or suggestion.input.get("as")
            if not name:
                raise ValueError("CSV_TO_VARIABLE requires a 'name' parameter to store the DataFrame")

            if not self._auto_confirm:
                path = self._prompt_for_path(path)

            if not path:
                raise ValueError("CSV_TO_VARIABLE requires a 'path' parameter")

            while True:
                try:
                    df = tool_fn(path)
                    break
                except FileNotFoundError as exc:
                    if self._auto_confirm:
                        raise
                    print(f"File not found: {path}")
                    path = self._prompt_for_path(None)
                    suggestion.input["path"] = path
                except Exception:
                    raise

            suggestion.input["path"] = path
            self._state.set_dataframe(name, df)
            return f"DataFrame '{name}' loaded from {path} with shape {df.shape}."

        if suggestion.tool == "SUMMARIZE_DATA":
            df_name = self._resolve_single_name(suggestion.input)
            df = self._state.get_dataframe(df_name)
            summary = tool_fn(df)
            return f"Summary for '{df_name}':\n{summary}"

        if suggestion.tool == "COMBINE_SUMMARIZE":
            names = self._resolve_name_list(suggestion.input)
            dfs = [self._state.get_dataframe(name) for name in names]
            summary = tool_fn(dfs)
            return "Combined summary:\n" + summary

        raise ValueError(f"Tool not yet implemented: {suggestion.tool}")

    @staticmethod
    def _resolve_single_name(payload: Dict[str, Any]) -> str:
        for key in ("df", "name", "target", "variable"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        raise ValueError("Expected a dataframe identifier in payload")

    @staticmethod
    def _resolve_name_list(payload: Dict[str, Any]) -> List[str]:
        for key in ("dfs", "inputs", "variables", "names"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                if all(isinstance(item, str) for item in value):
                    return value
        raise ValueError("Expected a non-empty list of dataframe identifiers")

    @staticmethod
    def _task_is_complete(result_text: str, suggestion: ToolSuggestion) -> bool:
        lowered = result_text.lower()
        if "task complete" in lowered or "no further action" in lowered:
            return True

        if suggestion.tool == "COMBINE_SUMMARIZE":
            desc = suggestion.description.lower()
            if "combined insight" in desc or "combined insights" in desc:
                return True
            if lowered.startswith("combined summary"):
                return True

        return False

    def _prompt_for_path(self, default: Optional[str]) -> str:
        base_dir = Path.cwd()
        default_value = default
        default_display: Optional[str] = None
        if default:
            default_path = Path(default)
            if default_path.is_absolute():
                try:
                    default_display = str(default_path.relative_to(base_dir))
                except ValueError:
                    default_display = str(default_path)
            else:
                default_display = default

        while True:
            prompt = f"Enter CSV path relative to {base_dir}"
            if default_display:
                prompt += f" [{default_display}]"
            prompt += ": "
            entered = input(prompt).strip()
            if not entered and default_value:
                return default_value
            if entered:
                return entered
            print("Path cannot be empty.")


def run_cli(
    task: Optional[str] = None,
    auto_confirm: bool = False,
    llm_client: Optional[LLMClient] = None,
) -> None:
    """Entry point used by the CLI wrapper."""
    if task is None:
        task = input("Enter your high-level task: ").strip()
    if not task:
        raise ValueError("A task description is required")

    agent = InteractiveAgent(llm_client=llm_client, auto_confirm=auto_confirm)
    agent.run(task)
