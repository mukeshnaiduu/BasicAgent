
"""LLM manager and prompt builder.

This module provides a small LLMManager that builds task-level and subtask-level
prompts using a base system prompt and a dynamic tools context. It includes a
simple local JSON-backed mock completion when no real LLM is configured. The
goal is to keep the prompt formatting and message assembly in one place so the
rest of the agent can remain LLM-agnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any


class LLMManager:
	def __init__(self, base_prompt_path: str | None = None, model: str | None = None):
		# model is kept for future extension; defaults can come from configs.settings
		self.model = model or os.environ.get("LLM_MODEL")
		base_path = base_prompt_path or "configs/base_prompt.txt"
		self.base_prompt = self._load_base_prompt(base_path)

	def _load_base_prompt(self, path: str) -> str:
		try:
			p = Path(path)
			if p.exists():
				return p.read_text(encoding="utf-8")
		except Exception:
			pass
		# sensible default if no file present
		return (
			"You are an analytics planning assistant. You will be given a task and a list"
			" of available tools (name, purpose, parameters). Produce a JSON plan that lists"
			" sequential steps. Each step must specify the tool name and a params object."
		)

	def build_tools_context(self, tools_info: Dict[str, Dict[str, Any]]) -> str:
		"""Serialize the tools info into a compact text block for inclusion in prompts."""
		lines: List[str] = ["Available tools:"]
		for key, info in sorted(tools_info.items()):
			params = info.get("parameters") or {}
			param_list = []
			for pname, pmeta in params.items():
				if isinstance(pmeta, dict):
					desc = pmeta.get("description", "")
					param_list.append(f"{pname}: {desc}")
				else:
					param_list.append(f"{pname}: {pmeta}")
			lines.append(f"- {key}: {info.get('purpose', '')}")
			if param_list:
				lines.append("  params: " + ", ".join(param_list))
		return "\n".join(lines)

	def build_task_prompt(self, task: str, tools_info: Dict[str, Dict[str, Any]], history: List[Dict[str, Any]] | None = None) -> str:
		"""Build the full task-level prompt string combining base prompt, tools and history.

		The returned string is intended to be sent to the LLM as the user-facing
		instruction asking for a structured JSON plan.
		"""
		history = history or []
		parts: List[str] = [self.base_prompt, "", self.build_tools_context(tools_info), ""]
		parts.append("Instructions: Return a JSON object with the shape:\n{\n  \"steps\": [ {\"id\":..., \"tool\":..., \"description\":..., \"params\": {...} } ], \n  \"notes\": \"optional\"\n}")
		parts.append("")
		if history:
			parts.append("Previous conversation and actions:")
			for item in history:
				# history items expected to be dicts with role/message
				role = item.get("role")
				content = item.get("content")
				parts.append(f"[{role}] {content}")
			parts.append("")
		parts.append(f"User task: {task}")
		return "\n\n".join(parts)

	def build_subtask_prompt(self, tool_name: str, tool_info: Dict[str, Any], history: List[Dict[str, Any]] | None = None) -> str:
		"""Build a subtask prompt asking the LLM to provide parameters for a named tool.

		The model should respond with a JSON object containing the parameters only.
		"""
		history = history or []
		parts = [self.base_prompt, "", f"Tool: {tool_name}", f"Purpose: {tool_info.get('purpose')}", "Parameters schema:", json.dumps(tool_info.get("parameters", {}), indent=2)]
		if history:
			parts.append("Context:")
			for h in history:
				parts.append(f"[{h.get('role')}] {h.get('content')}")
		parts.append("\nReturn a JSON object containing only the 'params' object with values for the tool.")
		return "\n\n".join(parts)

	def json_completion(self, prompt: str) -> Dict[str, Any]:
		raise NotImplementedError("LLM completion must be provided by an external llm_client. Use llm_client.call(prompt) in the agent.")


