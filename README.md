# BasicAgent

Interactive AI agent workflow where the model proposes tool executions and the user chooses which "case" to run next.

## Overview

This project packages a lightweight command-line agent that:

- Receives a high-level task from the user (e.g., "Analyze CSV files in folder").
- Asks an LLM to plan the next step using a fixed base prompt and available tools.
- Shows each suggested subtask to the user and waits for confirmation before executing it.
- Executes tool functions (`CSV_TO_VARIABLE`, `SUMMARIZE_DATA`, `COMBINE_SUMMARIZE`) when approved and feeds the results back into the conversation loop.

The workflow keeps the user in control while still leveraging the LLM to plan complex data-analysis sequences.

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

Launch the Streamlit frontend for a button-driven experience:

```bash
streamlit run streamlit_app.py
```

Workflow tips:
- Enter your Gemini API key in the sidebar (stored only in the current session).
- If the key already lives in `.env`, the app loads it automatically—no need to re-enter unless you want to override it for the session.
- Provide an optional model override or enable auto-confirmation from the same panel.
- Describe a task in the main view; the app requests the first suggestion automatically.
- Review each proposed tool call, adjust CSV paths when needed, and run or reject the step.
- The conversation history and loaded DataFrames remain visible for reference.

Sample interaction:

```
Starting interactive workflow for task: Analyze CSV files in data/

--- Iteration 1 ---
LLM Suggestion:
Tool: CSV_TO_VARIABLE
Input: {
	"path": "data/customers.csv",
	"name": "df_customers"
}
Description: Load customers.csv into df_customers
Run this step? (y/n/exit): y
DataFrame 'df_customers' loaded from data/customers.csv with shape (1200, 12).

--- Iteration 2 ---
LLM Suggestion:
Tool: SUMMARIZE_DATA
Input: {
	"df": "df_customers"
}
Description: Summarize df_customers to understand key metrics
Run this step? (y/n/exit): y
Summary for 'df_customers':
...
```

Declining a step (`n`) prompts the LLM for an alternate plan. Type `exit` at any prompt to stop the workflow.
When loading CSVs, you'll be prompted for the file path relative to the project root—press Enter to accept the suggestion or type a new relative/absolute location. The value you choose is echoed back into the workflow so downstream steps refer to the correct DataFrame.

To run without confirmations (useful for demos), append `--auto-confirm`.

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