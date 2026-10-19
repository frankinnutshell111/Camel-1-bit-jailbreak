from __future__ import annotations

from types import SimpleNamespace

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    text_content_block_from_string,
)

from agentic_red_teaming.evaluate_trajectory import (
    PlanJudgeResult,
    evaluate_trajectory,
    extract_pllm_plans,
)
from agentic_red_teaming.graph import (
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
    Trajectory,
)


def _text(value: str) -> list:
    return [text_content_block_from_string(value)]


def _messages() -> tuple[ChatMessage, ...]:
    tool_call = FunctionCall(function="get_unread_emails", args={})
    return (
        ChatUserMessage(role="user", content=_text("Original query")),
        ChatToolResultMessage(
            role="tool",
            tool_call=tool_call,
            content=_text("untrusted result"),
            tool_call_id=None,
            error=None,
        ),
        ChatAssistantMessage(
            role="assistant",
            content=_text("failed_plan()"),
            tool_calls=None,
        ),
        ChatUserMessage(
            role="user",
            content=_text("Running the code gave the following error:\nNotEnoughInformationError"),
        ),
        ChatAssistantMessage(
            role="assistant",
            content=_text("successful_plan()"),
            tool_calls=[],
        ),
        ChatAssistantMessage(
            role="assistant",
            content=_text("Finished user-facing answer"),
            tool_calls=None,
        ),
    )


def _config() -> RedTeamingConfig:
    return RedTeamingConfig(
        difficulty=TaskGenerationDifficulty.MEDIUM,
        injection_mode=InjectionSelectionMode.SINGLE,
        num_retries=1,
        evaluator_model="test:evaluator",
    )


def test_extract_pllm_plans_handles_failed_and_successful_attempts() -> None:
    assert extract_pllm_plans(_messages()) == (
        "failed_plan()",
        "successful_plan()",
    )


def test_evaluate_trajectory_aggregates_any_jailbreak(monkeypatch) -> None:
    judgments = iter(
        (
            PlanJudgeResult(
                jailbreak=False,
                dramatic_deviation=False,
                unsafe_risk=False,
                explanation="The plan remains within the requested task.",
            ),
            PlanJudgeResult(
                jailbreak=True,
                dramatic_deviation=True,
                unsafe_risk=True,
                explanation="The plan sends unrelated private data externally.",
            ),
        )
    )
    prompts: list[str] = []

    class FakeAgent:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run_sync(self, prompt: str) -> SimpleNamespace:
            prompts.append(prompt)
            return SimpleNamespace(data=next(judgments))

    monkeypatch.setattr(
        "agentic_red_teaming.evaluate_trajectory.pydantic_ai.Agent",
        FakeAgent,
    )
    trajectory = Trajectory(
        trajectory_id="trajectory_test",
        retry_index=0,
        messages=_messages(),
        model_output="Finished user-facing answer",
        tool_calls=(),
        task_completed=None,
    )

    evaluation = evaluate_trajectory(trajectory, "Original query", _config())

    assert len(prompts) == 2
    assert evaluation.jailbreak is True
    assert evaluation.deviated_from_task is True
    assert evaluation.unsafe_risk is True
    assert evaluation.explanations == (
        "Plan 1: The plan remains within the requested task.",
        "Plan 2: The plan sends unrelated private data externally.",
    )
    assert len(evaluation.plan_evaluations) == 2
    assert evaluation.metadata["plan_count"] == 2


def test_evaluate_trajectory_without_plans_does_not_call_judge(monkeypatch) -> None:
    class UnexpectedAgent:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("The evaluator should not be created without plans")

    monkeypatch.setattr(
        "agentic_red_teaming.evaluate_trajectory.pydantic_ai.Agent",
        UnexpectedAgent,
    )
    trajectory = Trajectory(
        trajectory_id="trajectory_empty",
        retry_index=0,
        messages=(
            ChatUserMessage(role="user", content=_text("Original query")),
            ChatAssistantMessage(
                role="assistant",
                content=_text("Final answer only"),
                tool_calls=None,
            ),
        ),
        model_output="Final answer only",
        tool_calls=(),
        task_completed=None,
    )

    evaluation = evaluate_trajectory(trajectory, "Original query", _config())

    assert evaluation.jailbreak is False
    assert evaluation.explanations == ()
    assert evaluation.metadata["plan_count"] == 0
