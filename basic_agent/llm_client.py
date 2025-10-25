from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List


from .prompts import BASE_PROMPT
from .types import ToolSuggestion


class LLMClientError(RuntimeError):
    """Raised when the LLM client cannot produce a valid suggestion."""


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMClient:
    """Gemini-powered client that returns structured tool suggestions."""

    def __init__(self, model: str | None = None, base_prompt: str = BASE_PROMPT) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is required")

        try:
            from google import generativeai as genai
        except ImportError as exc:  # pragma: no cover - missing optional dependency
            raise ImportError("Install 'google-generativeai' to use the Gemini backend.") from exc

        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        self._genai = genai
        raw_model = model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        legacy_aliases = {
            "gemini-1.5-flash": "gemini-flash-latest",
            "gemini-1.5-flash-latest": "gemini-flash-latest",
            "gemini-1.5-pro": "gemini-pro-latest",
            "gemini-1.5-pro-latest": "gemini-pro-latest",
            "gemini-2.0-flash-latest": "gemini-2.0-flash",
        }
        raw_model = legacy_aliases.get(raw_model, raw_model)
        self._model_name = raw_model
        self._client = genai.GenerativeModel(self._model_name)
        self._system_prompt = base_prompt

    def suggestion(self, history: Iterable[ChatMessage]) -> ToolSuggestion:
        prompt = self._build_prompt(history)

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            content = getattr(response, "text", None)
            if not content and hasattr(response, "candidates"):
                content = self._extract_first_text(response)
        except Exception as exc:  # pragma: no cover - network failures
            raise LLMClientError(f"Failed to obtain suggestion from Gemini: {exc}") from exc

        parsed = self._ensure_json(content or "")
        try:
            return ToolSuggestion.from_json(parsed)
        except Exception as exc:
            # Provide the raw content in the error so the UI can show helpful debug info
            raw = content or ""
            raise LLMClientError(f"LLM returned an invalid suggestion payload: {exc}; raw_response={raw}") from exc

    def json_completion(self, prompt: str, *, include_base_prompt: bool = False) -> dict:
        """Request a structured JSON response from the model using the provided prompt."""

        if include_base_prompt:
            composed_prompt = f"{self._system_prompt}\n\n{prompt}".strip()
        else:
            composed_prompt = prompt

        try:
            response = self._client.generate_content(
                composed_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            content = getattr(response, "text", None)
            if not content and hasattr(response, "candidates"):
                content = self._extract_first_text(response)
        except Exception as exc:  # pragma: no cover - network failures or API issues
            raise LLMClientError(f"Failed to obtain JSON completion from Gemini: {exc}") from exc

        try:
            return self._ensure_json(content or "")
        except LLMClientError as exc:
            raw = content or ""
            raise LLMClientError(f"Gemini returned non-JSON content: {raw}") from exc

    def _build_prompt(self, history: Iterable[ChatMessage]) -> str:
        lines: List[str] = [self._system_prompt, ""]
        for message in history:
            role = message.role.upper()
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_first_text(response: object) -> str:
        try:
            candidate = response.candidates[0]  # type: ignore[index]
            part = candidate.content.parts[0]
            return getattr(part, "text", "")
        except Exception as exc:  # pragma: no cover - structure differences
            raise LLMClientError("Unexpected response format from Gemini") from exc

    @staticmethod
    def _ensure_json(raw: str) -> dict:
        raw = raw.strip()
        if not raw:
            raise LLMClientError("LLM returned an empty response")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response was not valid JSON") from exc
