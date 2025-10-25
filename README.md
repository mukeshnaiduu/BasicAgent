# BasicAgent

Interactive AI agent workflow that can run as a CLI or through a guided Streamlit interface. The Streamlit UI now walks through the full process: capture the task prompt, gather/upload CSVs, review summaries, let the LLM design join steps, and present combined insights.

## Overview

This project packages two complementary experiences:

- **CLI agent** – a lightweight command-line orchestrator that breaks work into LLM-proposed tool calls you approve step-by-step.
- **Streamlit workflow** – a richer, stage-based UI that covers the exact flow described in the requirements: capture the prompt, surface expected files, upload/select CSVs, summarize each dataset, let the LLM choose join keys, and deliver combined insights plus metadata.

## Project Structure

```
basic_agent/
	agent.py          # Core interactive loop and CLI utilities
	llm_client.py     # Gemini LLM client used by the workflow
	prompts.py        # Base prompt the LLM receives on every call
	tools.py          # Implementations of supported data tools
	types.py          # Dataclasses shared across modules
main.py             # Command-line entry point
requirements.txt    # Runtime and test dependencies
tests/              # Lightweight test coverage for tools
```

## Base Prompt

The agent always primes the LLM with this instruction block:

```
You are an AI agent that performs tasks using the following tools:
1) CSV_TO_VARIABLE(path) -> returns DataFrame
2) SUMMARIZE_DATA(df) -> returns insights
3) COMBINE_SUMMARIZE([df1, df2, ...]) -> returns combined insights

Instructions:
- Break user tasks into subtasks.
- For each subtask, specify:
	 a) Tool to use
	 b) Input parameters
	 c) Short description of action
- Respond in JSON format:
{
	"tool": "<tool_name>",
	"input": {...},
	"description": "<short description of action>"
}
- Wait for user confirmation before executing each step.
- Continue until task is fully completed.
```

## Tool Registry

| Tool | Description | Expected Input Keys | Output |
| --- | --- | --- | --- |
| `CSV_TO_VARIABLE` | Load a CSV file into pandas and store it in memory. | `path`, `name` | A DataFrame stored under `name`. | 
| `SUMMARIZE_DATA` | Summarise a single DataFrame. | `df` / `name` / `target` | Text summary string. |
| `COMBINE_SUMMARIZE` | Concatenate multiple DataFrames and summarise. | `dfs` / `names` / `variables` | Text summary string. |

Suggestions must reference the stored DataFrame names so the agent can retrieve them.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API credentials before running the agent (copy `.env.example` to `.env`; the CLI automatically loads it via `python-dotenv`):

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-2.0-flash"  # optional override
```

## Usage

Run the CLI and follow the prompts:

```bash
python main.py "Analyze CSV files in data/"
```

Override the model on demand without touching environment variables:

```bash
python main.py "Analyze CSV files" --model gemini-flash-latest
```

## Streamlit UI

Launch the Streamlit frontend for the guided, multi-step workflow:

```bash
streamlit run streamlit_app.py
```

Workflow tips:
- Enter your Gemini API key in the sidebar (stored only in the current session). If the key already lives in `.env`, the app loads it automatically.
- Step 1 – Describe your task. The LLM returns a JSON checklist of expected files/notes.
- Step 2 – Upload CSVs or pick them from a local folder; the UI tracks both sources with a unified checklist.
- Step 3 – Load every selected CSV into a pandas DataFrame, with summaries and previews rendered for each.
- Step 4 – Ask the LLM to recommend join keys and join order. The app executes the plan, shows the join steps that ran, and produces a combined summary.
- Step 5 – Review the final dataset preview, combined summary, and extra insights. Metadata (including join details) is saved to `data/csv_relations.json` for downstream automation.

During the workflow, the UI surfaces structured feedback (LLM recommendations, join plan, insights) and writes an updated `data/csv_relations.json` containing file metadata, join plan, and combined summary so subsequent runs or automations can reuse the context.

## Implementing Custom Tools

Extend `basic_agent/tools.py` with new functions and register them in `TOOL_REGISTRY`. Make sure the LLM prompt is updated to document the new capability, then retrain or fine-tune your prompt examples so the model knows how to call it.

## Running Tests

```bash
pytest
```

The current test suite covers the data summarisation helpers. Expand it as you add new tools or behaviours.

## Next Steps

- Add richer error handling for malformed JSON suggestions.
- Cache tool outputs (e.g., saving DataFrames to disk) to persist state across sessions.
- Provide prompt templates or few-shot examples so the LLM learns preferred variable names and tool arguments.