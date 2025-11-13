"""Application settings and loader.

This module centralizes small runtime settings used by the agent and LLM
integration. It reads environment variables with sensible defaults and
exposes a `Settings` dataclass and `load_settings()` helper.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
	gemini_api_key: Optional[str]
	gemini_model: str
	base_prompt_path: Optional[str]
	default_temperature: float
	ollama_base_url: str
	ollama_model: str


def load_settings() -> Settings:
	"""Load settings from environment and repo layout.

	- GEMINI_API_KEY: API key for Gemini (optional, can be set in .env)
	- GEMINI_MODEL: model id (default: gemini-1.5-flash)
	- BASE_PROMPT_PATH: optional path to a base prompt file
	- OLLAMA_BASE_URL: base URL for the local Ollama server (default: http://localhost:11434)
	- OLLAMA_MODEL: model name to request from Ollama (default: llama3.1)
	"""
	gemini_api_key = os.environ.get("GEMINI_API_KEY")
	gemini_model = os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash"
	base_prompt_path = os.environ.get("BASE_PROMPT_PATH")
	if not base_prompt_path:
		# look for a repository-local prompt file
		candidate = Path("configs") / "base_prompt.txt"
		if candidate.exists():
			base_prompt_path = str(candidate)

	ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
	ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")

	return Settings(
		gemini_api_key,
		gemini_model,
		base_prompt_path,
		float(os.environ.get("DEFAULT_TEMPERATURE", "0.0")),
		ollama_base_url,
		ollama_model,
	)

