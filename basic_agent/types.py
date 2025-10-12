from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolSuggestion:
    tool: str
    input: Dict[str, Any]
    description: str

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "ToolSuggestion":
        try:
            tool = payload["tool"].strip()
            input_payload = payload.get("input") or {}
            description = payload.get("description", "").strip()
        except (KeyError, AttributeError) as exc:
            raise ValueError("Invalid suggestion payload") from exc

        if not tool:
            raise ValueError("Tool name is required in suggestion")

        if not isinstance(input_payload, dict):
            raise ValueError("Suggestion 'input' must be a JSON object")

        if not description:
            raise ValueError("Suggestion 'description' must be provided")

        return cls(tool=tool.upper(), input=input_payload, description=description)
