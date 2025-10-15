from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from basic_agent.agent import run_cli
from basic_agent.llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive AI agent with user-selected tool execution.")
    parser.add_argument("task", nargs="?", help="High-level task description for the agent.")
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Execute suggested steps without prompting for confirmation (useful for demos).",
    )
    parser.add_argument(
        "--model",
        help="Override the Gemini model name (optional).",
    )
    return parser


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    parser = build_parser()
    args = parser.parse_args()
    llm_client = LLMClient(model=args.model)
    run_cli(task=args.task, auto_confirm=args.auto_confirm, llm_client=llm_client)


if __name__ == "__main__":
    main()
