from basic_agent.agent import InteractiveAgent
from basic_agent.types import ToolSuggestion


def test_task_completes_after_combined_insights_description() -> None:
    suggestion = ToolSuggestion(
        tool="COMBINE_SUMMARIZE",
        input={},
        description="Generate combined insights across customer, order, and product data.",
    )
    assert InteractiveAgent._task_is_complete("Combined summary:\nRows: 10", suggestion)


def test_task_completes_only_on_keywords() -> None:
    non_terminal_suggestion = ToolSuggestion(
        tool="SUMMARIZE_DATA",
        input={},
        description="Summarize customer data",
    )
    assert not InteractiveAgent._task_is_complete("Summary for customers", non_terminal_suggestion)
