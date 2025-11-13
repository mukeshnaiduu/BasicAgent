"""Task plan parsing and validation helpers.

This module provides a small set of helpers to validate an LLM-produced plan
against a tools registry. It intentionally avoids importing tool modules
automatically (some environments may have tools that are not importable at
development time). Instead callers should pass a `tools_registry` mapping where
keys are tool names and values are the TOOL_INFO dicts exposed by the tool
modules.

Expected plan shape (validated):
{
  "steps": [ {"id": "step_1", "tool": "csv_loader", "description": "...", "params": {...} }, ... ],
  "notes": "..."
}
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def normalize_plan(raw: Dict[str, Any]) -> Dict[str, Any]:
	"""Return a normalized plan dict with at least a 'steps' list.

	This function is permissive: it will convert a single-step dict into the
	expected shape when necessary.
	"""
	if not isinstance(raw, dict):
		raise TypeError("Plan must be a dict-like JSON object")
	steps = raw.get("steps")
	if steps is None:
		# allow single-step plans where the root is a step
		if all(k in raw for k in ("tool", "params")):
			steps = [raw]
		else:
			steps = []
	if not isinstance(steps, list):
		raise TypeError("Plan 'steps' must be a list")
	return {"steps": steps, "notes": raw.get("notes")}


def validate_plan(plan: Dict[str, Any], tools_registry: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
	"""Validate an LLM-produced plan against the provided tools registry.

	Returns a tuple: (is_valid, errors, normalized_steps).

	- is_valid: True if no validation errors were found.
	- errors: list of human-readable error messages.
	- normalized_steps: list of steps (dict) ready for execution.
	"""
	errors: List[str] = []
	normalized = normalize_plan(plan)
	steps: List[Dict[str, Any]] = normalized.get("steps", [])

	out_steps: List[Dict[str, Any]] = []
	for idx, step in enumerate(steps, start=1):
		if not isinstance(step, dict):
			errors.append(f"step {idx} is not an object")
			continue
		tool = step.get("tool")
		if not tool:
			errors.append(f"step {idx} missing 'tool' field")
			continue
		if tool not in tools_registry:
			errors.append(f"step {idx}: unknown tool '{tool}'")
			continue
		tool_info = tools_registry[tool]
		params = step.get("params") or {}
		if not isinstance(params, dict):
			errors.append(f"step {idx}: 'params' must be an object/dict")
			continue

		# check required parameters
		required = [k for k, v in (tool_info.get("parameters") or {}).items() if isinstance(v, dict) and v.get("required")]
		missing = [r for r in required if r not in params]
		if missing:
			errors.append(f"step {idx} ({tool}): missing required params: {missing}")

		out_steps.append({
			"id": step.get("id") or f"step_{idx}",
			"tool": tool,
			"description": step.get("description"),
			"params": params,
		})

	is_valid = len(errors) == 0
	return is_valid, errors, out_steps


def step_to_history_entry(step: Dict[str, Any], result: Any) -> Dict[str, Any]:
	"""Create a compact history entry from an executed step and its result.

	This is intended to be appended to the conversation history that will be
	forwarded back to the LLM when generating subsequent subtasks.
	"""
	return {"role": "tool", "content": f"Executed {step.get('id')} ({step.get('tool')}), result: {repr(result)}"}



