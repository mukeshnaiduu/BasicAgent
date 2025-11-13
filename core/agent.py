"""Core agent orchestration.

This module exposes a lightweight Agent class that:
- discovers available tools under the `tools` package (imports them safely),
- builds task and subtask prompts via `LLMManager`,
- calls an external `llm_client(prompt: str) -> str` for completions (caller must
  provide a function that returns the LLM's text response),
- validates the top-level plan using `task_parser`,
- executes each step by resolving DataFrame ids in a simple in-memory session
  and invoking the discovered tool callables.

Design decisions:
- The agent does not try to be smart about tool function names. For each tool
  module it selects the first public callable attribute it finds (excluding
  `TOOL_INFO`). This keeps the discovery simple; the tools in /tools are
  expected to expose a single clear entrypoint.
- The external LLM client is required: pass a callable `llm_client(prompt: str)`
  that returns the LLM response as a string (typically JSON). The agent will
  parse JSON responses where appropriate.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from pathlib import Path
from copy import deepcopy
import requests

from .llm_manager import LLMManager
from . import task_parser
from configs.settings import load_settings
import os
import logging


# ensure logs directory exists and configure logger
LOG_DIR = os.environ.get("BASICAGENT_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "agent.log")
logger = logging.getLogger("basicagent.agent")
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


class ToolSpec:
    def __init__(self, name: str, module: ModuleType, info: Dict[str, Any], fn: Callable[..., Any]):
        self.name = name
        self.module = module
        self.info = info
        self.fn = fn


class Agent:
    def __init__(self, tools_package: str = "tools", llm_manager: Optional[LLMManager] = None):
        self.tools_package = tools_package
        self.llm_manager = llm_manager or LLMManager()
        self.tools: Dict[str, ToolSpec] = {}

    def discover_tools(self) -> Dict[str, ToolSpec]:
        """Import modules from the tools package and collect TOOL_INFO and a callable.

        Returns mapping tool_name -> ToolSpec.
        """
        discovered: Dict[str, ToolSpec] = {}
        try:
            pkg = importlib.import_module(self.tools_package)
        except Exception as e:
            raise ImportError(f"Unable to import tools package '{self.tools_package}': {e}")

        # iterate all modules in the package
        prefix = pkg.__name__ + "."
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix):
            try:
                mod = importlib.import_module(name)
            except Exception as e:
                # skip modules that fail to import
                continue
            info = getattr(mod, "TOOL_INFO", None)
            if not info or not isinstance(info, dict):
                continue

            tool_name = info.get("name") or mod.__name__.split(".")[-1]

            # find a callable in module to execute - exclude TOOL_INFO
            fn = None
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                if attr_name == "TOOL_INFO":
                    continue
                attr = getattr(mod, attr_name)
                if callable(attr) and inspect.isfunction(attr):
                    fn = attr
                    break

            if fn is None:
                # no callable found; skip
                continue

            discovered[tool_name] = ToolSpec(tool_name, mod, info, fn)

        self.tools = discovered
        return discovered

    def _parse_llm_json(self, text: str) -> Any:
        """Parse LLM response that is expected to be JSON.

        Tries to extract the first JSON object from the text. Raises ValueError
        when parsing fails.
        """
        text = text.strip()

        # quick direct parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # remove common markdown/code fences
        def _strip_code_fences(s: str) -> str:
            # remove ```json ... ``` or ``` ... ``` blocks and return inner content if it looks like JSON
            import re

            # try fenced code blocks first
            m = re.search(r"```(?:json\s*)?(.*?)```", s, flags=re.S | re.I)
            if m:
                inner = m.group(1).strip()
                if inner:
                    return inner
            # remove single-line ``` markers
            s = re.sub(r"```", "", s)
            # strip surrounding backticks
            s = s.strip('\n `')
            return s

        stripped = _strip_code_fences(text)
        try:
            return json.loads(stripped)
        except Exception:
            pass

        # try to extract the first balanced JSON object or array starting at first { or [
        def _extract_first_json(s: str) -> Optional[str]:
            for i, ch in enumerate(s):
                if ch not in "[{":
                    continue
                open_ch = ch
                close_ch = "]" if ch == "[" else "}"
                depth = 0
                for j in range(i, len(s)):
                    if s[j] == open_ch:
                        depth += 1
                    elif s[j] == close_ch:
                        depth -= 1
                        if depth == 0:
                            return s[i : j + 1]
            return None

        candidate = _extract_first_json(text)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # as last attempt, try to find a {...} pair using crude find
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            sub = text[start : end + 1]
            try:
                return json.loads(sub)
            except Exception:
                pass

        raise ValueError("Could not parse JSON from LLM response")

    def execute(self, task: str, llm_client: Callable[[str], str], uploaded_files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a high-level task.

        Args:
            task: natural language task description.
            llm_client: callable that accepts a prompt string and returns the LLM text response.
            uploaded_files: optional mapping name->file-like for use by csv_loader.

        Returns:
            dict with 'session' (mapping df_id->DataFrame-like), 'history' (list), and 'results'
        """
        if not self.tools:
            self.discover_tools()

        tools_info = {name: spec.info for name, spec in self.tools.items()}

        history: List[Dict[str, Any]] = []
        session: Dict[str, Any] = {}
        next_df_index = 1
        unused_upload_keys: List[str] = list(uploaded_files.keys()) if uploaded_files else []
        step_output_ids: Dict[str, str] = {}
        df_aliases: Dict[str, str] = {}
        csv_loader_spec = self.tools.get("csv_loader")

        def _extract_df_placeholder_index(name: str) -> Optional[int]:
            lower = name.lower()
            if not lower.startswith("df"):
                return None
            remainder = lower[2:]
            while remainder and remainder[0] in "_-":
                remainder = remainder[1:]
            if remainder and remainder.isdigit():
                try:
                    return int(remainder)
                except ValueError:
                    return None
            return None

        def _store_dataframe(
            result: Any,
            tool_name: str,
            resolved_params: Dict[str, Any],
            step_context: Optional[Dict[str, Any]],
            raw_params_for_alias: Optional[Dict[str, Any]] = None,
            extra_aliases: Optional[Iterable[str]] = None,
            step_order_alias: Optional[str] = None,
        ) -> Optional[str]:
            nonlocal next_df_index
            try:
                import pandas as _pd

                if isinstance(result, _pd.DataFrame):
                    df_id = f"df_{next_df_index}"
                    session[df_id] = result
                    next_df_index += 1

                    alias_candidates: Set[str] = {df_id, df_id.replace("_", ""), df_id.replace("_", "-")}

                    if step_order_alias:
                        alias_candidates.update(
                            {
                                step_order_alias,
                                f"{step_order_alias}_result",
                                f"df_{step_order_alias}",
                                f"df_{step_order_alias}_result",
                                step_order_alias.replace("_", ""),
                            }
                        )
                        step_output_ids[step_order_alias] = df_id
                        step_output_ids[f"{step_order_alias}_result"] = df_id
                        step_output_ids[f"df_{step_order_alias}"] = df_id

                    step_id = step_context.get("id") if isinstance(step_context, dict) else None
                    if step_id:
                        step_output_ids[step_id] = df_id
                        step_output_ids[f"{step_id}_result"] = df_id
                        step_output_ids[f"df_{step_id}"] = df_id
                        alias_candidates.update(
                            {
                                step_id,
                                step_id.replace("-", "_"),
                                step_id.replace("-", ""),
                                f"{step_id}_result",
                                f"df_{step_id}",
                                f"df_{step_id}_result",
                                f"df{step_id}",
                                f"{step_id}_df",
                                f"{step_id}df",
                            }
                        )
                    else:
                        step_output_ids[df_id] = df_id

                    if isinstance(raw_params_for_alias, dict):
                        name_hint = raw_params_for_alias.get("name")
                        if isinstance(name_hint, str) and name_hint:
                            alias_candidates.add(name_hint)
                        if tool_name == "combine":
                            original_dfs = raw_params_for_alias.get("dfs")
                            if isinstance(original_dfs, list) and all(isinstance(x, str) for x in original_dfs):
                                joined = "_".join(original_dfs)
                                alias_candidates.update(
                                    {
                                        joined,
                                        f"{joined}_combined",
                                        f"{joined}_merge",
                                        f"{joined}_merged",
                                        f"{joined}_result",
                                    }
                                )

                    if tool_name == "csv_loader":
                        alias_source = resolved_params.get("name") if isinstance(resolved_params, dict) else None
                        if isinstance(alias_source, str) and alias_source:
                            alias_candidates.add(alias_source)
                            try:
                                from pathlib import Path as _AliasPath

                                alias_candidates.add(_AliasPath(alias_source).stem)
                            except Exception:
                                pass
                        path_val = resolved_params.get("path") if isinstance(resolved_params, dict) else None
                        if isinstance(path_val, str) and path_val:
                            try:
                                from pathlib import Path as _Path

                                path_obj = _Path(path_val)
                                alias_candidates.update({path_obj.name, path_obj.stem})
                            except Exception:
                                alias_candidates.add(path_val)

                    if extra_aliases:
                        alias_candidates.update(alias for alias in extra_aliases if alias)

                    expanded_aliases: Set[str] = set()
                    for alias in alias_candidates:
                        if not alias:
                            continue
                        lower_alias = alias.lower()
                        expanded_aliases.add(lower_alias)
                        squashed = lower_alias.replace("_", "").replace("-", "")
                        if squashed:
                            expanded_aliases.add(squashed)
                        alnum = "".join(ch for ch in lower_alias if ch.isalnum())
                        if alnum:
                            expanded_aliases.add(alnum)
                        base = alnum or squashed or lower_alias
                        if base:
                            if not base.startswith("df"):
                                expanded_aliases.add(f"df_{base}")
                                expanded_aliases.add(f"df{base}")
                            if not base.endswith("df"):
                                expanded_aliases.add(f"{base}_df")
                                expanded_aliases.add(f"{base}df")

                    for alias in expanded_aliases:
                        df_aliases[alias] = df_id
                        step_output_ids.setdefault(alias, df_id)

                    logger.info("Stored DataFrame result as %s with shape %s", df_id, result.shape)
                    logger.debug("Alias map updated for %s with aliases (sample): %s", df_id, list(sorted(expanded_aliases))[:10])
                    return df_id
            except Exception:
                logger.debug("Result not a DataFrame or pandas unavailable for tool %s", tool_name)
            return None

        def _auto_materialize_placeholder(identifier: str) -> Optional[Any]:
            if not uploaded_files or not csv_loader_spec or not unused_upload_keys:
                return None
            index = _extract_df_placeholder_index(identifier)
            if index is None:
                return None

            key = None
            target_idx = index - 1
            if 0 <= target_idx < len(unused_upload_keys):
                key = unused_upload_keys.pop(target_idx)
            elif unused_upload_keys:
                key = unused_upload_keys.pop(0)

            if key is None:
                return None

            path_obj = uploaded_files.get(key)
            if path_obj is None:
                return None

            params = {"path": str(path_obj), "name": key}
            try:
                loaded_df = csv_loader_spec.fn(**params)
            except Exception:
                logger.exception("Auto-loading upload '%s' for placeholder %s failed", key, identifier)
                unused_upload_keys.insert(0, key)
                return None

            fake_step = {"id": f"auto_load_{key}", "tool": "csv_loader", "description": "Auto-loaded upload"}
            df_id = _store_dataframe(
                loaded_df,
                "csv_loader",
                params,
                fake_step,
                raw_params_for_alias=params,
                extra_aliases={identifier, key},
                step_order_alias=None,
            )
            if df_id:
                logger.info("Auto-loaded upload '%s' for placeholder %s -> %s", key, identifier, df_id)
                return session[df_id]

            unused_upload_keys.insert(0, key)
            return None

        logger.info("Starting agent.execute for task: %s", task)
        logger.debug("Uploaded files mapping: %s", {k: str(v) for k, v in (uploaded_files or {}).items()})

        # 1) request an initial (coarse) plan from the LLM
        task_prompt = self.llm_manager.build_task_prompt(task, tools_info, history)
        logger.debug("Task prompt sent to LLM: %s", task_prompt)
        task_resp_text = llm_client(task_prompt)
        logger.debug("Task response text from LLM: %s", task_resp_text)
        try:
            plan_json = self._parse_llm_json(task_resp_text)
        except ValueError:
            # Ask the model to re-output strict JSON once (helpful recovery)
            correction_prompt = (
                "The previous response was not valid JSON. Please re-output ONLY the JSON plan (no explanation). "
                "The plan must be a JSON object with a 'steps' array where each step has at least a 'tool' field. "
                "Available tools: "
                + ", ".join(sorted(tools_info.keys()))
                + ". The original model output was:\n\n"
                + task_resp_text
            )
            logger.debug("Asking LLM for corrected JSON plan. Prompt: %s", correction_prompt)
            corrected = llm_client(correction_prompt)
            logger.debug("Corrected plan response: %s", corrected)
            plan_json = self._parse_llm_json(corrected)

        # Normalize plan shapes: accept a list of steps, or a single-step dict.
        logger.debug("Parsed plan_json: %s", repr(plan_json))
        if not isinstance(plan_json, dict):
            if isinstance(plan_json, list):
                plan_json = {"steps": plan_json}
            else:
                # last-ditch re-request for a dict-shaped plan
                correction2 = (
                    "Please output a JSON object with a top-level 'steps' array describing the plan. "
                    "Each step should be an object with at least a 'tool' field. "
                    "Do not include any explanatory text. The original output was:\n\n"
                    + (task_resp_text if isinstance(task_resp_text, str) else str(task_resp_text))
                )
                logger.debug("Asking LLM for dict-shaped plan. Prompt: %s", correction2)
                corrected2 = llm_client(correction2)
                logger.debug("Corrected2 plan response: %s", corrected2)
                plan_json = self._parse_llm_json(corrected2)

        # If the model returned a single step (dict with 'tool'), wrap it
        if isinstance(plan_json, dict) and "steps" not in plan_json and "tool" in plan_json:
            plan_json = {"steps": [plan_json]}

        logger.info("Final plan JSON: %s", json.dumps(plan_json) if isinstance(plan_json, dict) else repr(plan_json))
        # validate plan shape against tools registry
        is_valid, errors, steps = task_parser.validate_plan(plan_json, tools_info)
        if not is_valid:
            # Allow plans that omit required params (we will request them per-step).
            # Only block on other validation errors (unknown tool, malformed step, etc.).
            non_param_errors = [e for e in errors if "missing required params" not in e]
            if non_param_errors:
                logger.error("Plan validation failed: %s", errors)
                raise ValueError(f"Plan validation failed: {errors}")
            logger.warning("Plan has missing required params; will request them per-step: %s", errors)

        results: List[Dict[str, Any]] = []
        plan_snapshot = deepcopy(steps)
        step_counter = 0

        # iterate steps
        for step in steps:
            step_counter += 1
            step_order_alias = f"step_{step_counter}"
            tool_name = step["tool"]
            spec = self.tools.get(tool_name)
            if spec is None:
                raise KeyError(f"Tool not found: {tool_name}")

            logger.info("Executing step: %s", step)

            # build a subtask prompt to ask the LLM to provide params for this tool
            subtask_prompt = self.llm_manager.build_subtask_prompt(tool_name, spec.info, history + [{"role": "user", "content": task}])
            logger.debug("Subtask prompt for tool %s: %s", tool_name, subtask_prompt)
            subtask_resp_text = llm_client(subtask_prompt)
            logger.debug("Subtask response for tool %s: %s", tool_name, subtask_resp_text)

            # parse JSON params; allow responses that are either {"params": {...}} or {...}
            try:
                parsed = self._parse_llm_json(subtask_resp_text)
            except ValueError:
                # Ask the model to re-output ONLY the JSON params once
                params_schema = spec.info.get("parameters") or {}
                correction_prompt = (
                    "The previous response was not valid JSON. Please re-output ONLY the JSON object containing the parameters for the tool '"
                    + tool_name
                    + "'. Do not include any explanatory text. Schema: "
                    + json.dumps(params_schema)
                    + "\n\nOriginal output:\n\n"
                    + subtask_resp_text
                )
                logger.debug("Requesting corrected params JSON for tool %s. Prompt: %s", tool_name, correction_prompt)
                corrected = llm_client(correction_prompt)
                logger.debug("Corrected params response for tool %s: %s", tool_name, corrected)
                parsed = self._parse_llm_json(corrected)
            if isinstance(parsed, dict) and "params" in parsed:
                params = parsed["params"]
            elif isinstance(parsed, dict):
                params = parsed
            else:
                raise ValueError("Subtask LLM response did not contain an object with params")

            if not isinstance(params, dict):
                raise ValueError("Tool parameters must be represented as a JSON object")

            plan_defaults_raw = step.get("params")
            plan_defaults = plan_defaults_raw if isinstance(plan_defaults_raw, dict) else {}
            if plan_defaults:
                merged_params = deepcopy(plan_defaults)
                for key, value in params.items():
                    merged_params[key] = value
                params = merged_params

            raw_params = deepcopy(params)

            logger.debug("Parsed params for tool %s: %s", tool_name, params)

            # resolve parameters: replace df ids or special placeholder with actual objects
            # deep-resolve parameter values: replace session df ids in strings inside lists/dicts
            def _resolve_value(val):
                if isinstance(val, str):
                    if val in session:
                        return session[val]
                    mapped = step_output_ids.get(val)
                    if mapped and mapped in session:
                        return session[mapped]
                    alias_id = df_aliases.get(val.lower())
                    if alias_id and alias_id in session:
                        return session[alias_id]
                    auto_df = _auto_materialize_placeholder(val)
                    if auto_df is not None:
                        return auto_df
                    if val.startswith("<df") and val.endswith(">") and session:
                        digits = "".join(ch for ch in val if ch.isdigit())
                        keys = list(session.keys())
                        if digits:
                            try:
                                idx = int(digits) - 1
                                if 0 <= idx < len(keys):
                                    return session[keys[idx]]
                            except ValueError:
                                pass
                        return session[keys[0]]
                if isinstance(val, list):
                    return [_resolve_value(x) for x in val]
                if isinstance(val, dict):
                    return {kk: _resolve_value(vv) for kk, vv in val.items()}
                return val

            resolved_params = {}
            for k, v in params.items():
                if k == "uploaded" and uploaded_files is not None:
                    # allow uploaded to be a list or single token
                    if isinstance(v, str) and v in uploaded_files:
                        resolved_params[k] = uploaded_files[v]
                    else:
                        resolved_params[k] = uploaded_files
                else:
                    resolved_params[k] = _resolve_value(v)

            # Additional coercion for csv_loader to map placeholders to real files
            if tool_name == "csv_loader":
                if uploaded_files:
                    path_val = resolved_params.get("path")
                    if isinstance(path_val, str):
                        if path_val in uploaded_files:
                            resolved_params["path"] = str(uploaded_files[path_val])
                            resolved_params.setdefault("name", path_val)
                            if path_val in unused_upload_keys:
                                unused_upload_keys.remove(path_val)
                        else:
                            try:
                                desired = Path(path_val).name
                            except Exception:
                                desired = path_val
                            for key, stored_path in uploaded_files.items():
                                try:
                                    if Path(key).name == desired:
                                        resolved_params["path"] = str(stored_path)
                                        resolved_params.setdefault("name", key)
                                        if key in unused_upload_keys:
                                            unused_upload_keys.remove(key)
                                        break
                                except Exception:
                                    continue

                    uploaded_param = resolved_params.get("uploaded")
                    if isinstance(uploaded_param, str) and uploaded_param in uploaded_files:
                        resolved_params["path"] = str(uploaded_files[uploaded_param])
                        resolved_params.setdefault("name", uploaded_param)
                        resolved_params.pop("uploaded", None)
                        if uploaded_param in unused_upload_keys:
                            unused_upload_keys.remove(uploaded_param)
                    elif uploaded_param is not None:
                        # When the LLM returns an object/dict for `uploaded`, treat it as missing
                        resolved_params.pop("uploaded", None)
                        if unused_upload_keys:
                            fallback_key = unused_upload_keys.pop(0)
                            resolved_params["path"] = str(uploaded_files[fallback_key])
                            resolved_params.setdefault("name", fallback_key)

                    if not resolved_params.get("path") and "uploaded" not in resolved_params and unused_upload_keys:
                        fallback_key = unused_upload_keys.pop(0)
                        resolved_params["path"] = str(uploaded_files[fallback_key])
                        resolved_params.setdefault("name", fallback_key)

                if not resolved_params.get("path") and "uploaded" not in resolved_params:
                    raise ValueError("csv_loader requires a 'path' or 'uploaded' parameter")

            if tool_name == "combine":
                dfs_list = resolved_params.get("dfs")
                if dfs_list is None and isinstance(raw_params, dict) and raw_params.get("dfs") is not None:
                    resolved_params["dfs"] = _resolve_value(raw_params["dfs"])
                    dfs_list = resolved_params.get("dfs")
                if dfs_list is None:
                    raise ValueError("combine tool requires a 'dfs' parameter listing the DataFrames to merge")
                if isinstance(dfs_list, str):
                    resolved_params["dfs"] = [_resolve_value(dfs_list)]
                    dfs_list = resolved_params.get("dfs")
                if isinstance(dfs_list, list) and not dfs_list:
                    raise ValueError("combine tool requires at least one DataFrame in 'dfs'")
                if isinstance(dfs_list, list):
                    materialized = []
                    for item in dfs_list:
                        if isinstance(item, str):
                            candidate = session.get(item)
                            if candidate is None:
                                mapped_id = step_output_ids.get(item)
                                if mapped_id:
                                    candidate = session.get(mapped_id)
                            if candidate is None:
                                alias_id = df_aliases.get(item.lower())
                                if alias_id:
                                    candidate = session.get(alias_id)
                            if candidate is None:
                                raise ValueError(
                                    f"Unable to resolve DataFrame reference '{item}' for combine step. "
                                    "Ensure previous steps load the data before combining."
                                )
                            materialized.append(candidate)
                        else:
                            materialized.append(item)
                    resolved_params["dfs"] = materialized
                on_param = resolved_params.get("on")
                if isinstance(on_param, str) and on_param:
                    resolved_params["on"] = [on_param]

            logger.debug("Resolved params for tool %s: %s", tool_name, resolved_params)

            # call the tool function with some heuristics based on its signature
            sig = inspect.signature(spec.fn)
            fn_params = list(sig.parameters.keys())
            try:
                # If function accepts the given parameter names, call with kwargs
                if all((p in fn_params) for p in resolved_params.keys()):
                    result = spec.fn(**resolved_params)
                else:
                    # If function has a single parameter, try to pass the most
                    # relevant value (uploaded/path/dfs) or the whole params dict
                    if len(fn_params) == 1:
                        first_param = fn_params[0]
                        if first_param in resolved_params:
                            result = spec.fn(resolved_params[first_param])
                        else:
                            # try common keys
                            for candidate in ("uploaded", "path", "dfs", "df"):
                                if candidate in resolved_params:
                                    result = spec.fn(resolved_params[candidate])
                                    break
                            else:
                                result = spec.fn(resolved_params)
                    else:
                        # last resort: try kwargs and let Python raise if incompatible
                        result = spec.fn(**resolved_params)
            except FileNotFoundError as missing:
                if tool_name == "csv_loader" and unused_upload_keys and uploaded_files:
                    fallback_key = unused_upload_keys.pop(0)
                    fallback_path = uploaded_files.get(fallback_key)
                    if fallback_path is not None:
                        resolved_params["path"] = str(fallback_path)
                        resolved_params.setdefault("name", fallback_key)
                        logger.warning(
                            "csv_loader path '%s' not found. Falling back to uploaded '%s'", resolved_params.get("path"), fallback_key
                        )
                        try:
                            result = spec.fn(**resolved_params)
                        except Exception:
                            unused_upload_keys.insert(0, fallback_key)
                            raise
                    else:
                        unused_upload_keys.insert(0, fallback_key)
                        raise missing
                else:
                    raise
            except TypeError:
                # final fallback: call with single dict arg
                result = spec.fn(resolved_params)

            df_id = _store_dataframe(
                result,
                tool_name,
                resolved_params,
                step,
                raw_params_for_alias=raw_params,
                extra_aliases=None,
                step_order_alias=step_order_alias,
            )

            # append step result to results and history
            step_summary = {
                "id": step.get("id"),
                "tool": tool_name,
                "description": step.get("description"),
                "params": raw_params,
            }

            display_payload: Dict[str, Any]
            try:
                import pandas as _pd_display

                if isinstance(result, _pd_display.DataFrame):
                    preview = result.head(50)
                    display_payload = {
                        "type": "dataframe",
                        "preview": preview,
                        "shape": [int(result.shape[0]), int(result.shape[1])],
                        "columns": list(result.columns),
                    }
                elif isinstance(result, dict):
                    display_payload = {"type": "dict", "content": result}
                else:
                    display_payload = {"type": "text", "content": str(result)}
            except Exception:
                display_payload = {"type": "text", "content": repr(result)}

            results.append({
                "step": step_summary,
                "result": display_payload,
                "raw": repr(result),
            })
            history_entry = task_parser.step_to_history_entry(step, result)
            history.append(history_entry)
            logger.info("Step completed. tool=%s result=%s", tool_name, repr(result))
            logger.debug("Appended history entry: %s", history_entry)

        logger.info("Agent execution finished. session size=%d, steps=%d", len(session), len(results))
        logger.debug("Final session keys: %s", list(session.keys()))
        return {"session": session, "history": history, "results": results, "plan": plan_snapshot}


def simple_llm_client_from_gemini(api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0) -> Callable[[str], str]:
    """Simple Gemini client wrapper (library-only).

    This function assumes the `google.generativeai` library is installed and
    available as the global `genai` symbol. It configures the library with the
    provided API key (or value from settings) and returns a callable that takes
    a prompt string and returns the model's text output.

    This implementation intentionally keeps a single, minimal code path and
    avoids trying multiple compatibility fallbacks. If the library or
    credentials are not available this will raise the underlying exception.
    """
    # Use centralized settings as the single source of defaults.
    settings = load_settings()
    api_key = api_key or settings.gemini_api_key
    model = model or settings.gemini_model or "gemini-1.5"

    # Ensure the global google.generativeai is importable
    try:
        # prefer a globally-imported genai if present
        gen = globals().get("genai")
        if gen is None:
            import google.generativeai as genai  # type: ignore
            gen = genai
            globals()["genai"] = gen
    except Exception:
        logger.exception("google.generativeai import failed")
        raise

    # configure API key
    gen.configure(api_key=api_key)

    # construct a GenerativeModel instance for the requested model name
    model_inst = gen.GenerativeModel(model)

    def _client(prompt_text: str) -> str:
        # Call the model with a plain string prompt. Use generation_config to
        # set temperature if provided.
        gen_cfg = {"temperature": float(temperature)} if temperature is not None else None
        resp = model_inst.generate_content(prompt_text, generation_config=gen_cfg)

        # extract textual output from common shapes
        if hasattr(resp, "text") and resp.text:
            return resp.text
        if getattr(resp, "candidates", None):
            try:
                cand = resp.candidates[0]
                return getattr(cand, "output", None) or getattr(cand, "content", None) or str(cand)
            except Exception:
                pass
        return str(resp)

    return _client


def simple_llm_client_from_ollama(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Callable[[str], str]:
    """Simple client that talks to a local Ollama server (e.g., llama3.1)."""

    settings = load_settings()
    model_name = model or getattr(settings, "ollama_model", None) or "llama3.1"
    base = base_url or getattr(settings, "ollama_base_url", None) or "http://localhost:11434"
    endpoint = base.rstrip("/") + "/api/generate"

    def _client(prompt_text: str) -> str:
        payload: Dict[str, Any] = {"model": model_name, "prompt": prompt_text, "stream": False}
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network errors are runtime concerns
            logger.exception("Ollama request failed")
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        text = data.get("response")
        if text is None:
            fallback = data.get("data") or data.get("message")
            if isinstance(fallback, list):
                text = "".join(str(chunk) for chunk in fallback)
            elif fallback is not None:
                text = str(fallback)
        if text is None:
            text = str(data)
        return text

    return _client


