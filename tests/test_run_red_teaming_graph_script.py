from __future__ import annotations

import pytest

from agentic_red_teaming.graph import (
    InjectionSelectionMode,
    TaskGenerationDifficulty,
)
from scripts.run_red_teaming_graph import _build_config, _parse_args


def test_cli_options_build_red_teaming_config() -> None:
    args = _parse_args(
        [
            "--log-folder",
            "logs/test-run",
            "--number-of-tasks",
            "4",
            "--sampling-size",
            "7",
            "--pllm",
            "openai:gpt-5",
            "--pllm-retry-budget",
            "12",
            "--qllm",
            "openai:gpt-4.1",
            "--task-generator-llm",
            "openai:gpt-5-mini",
            "--evaluation-llm",
            "openai:gpt-4.1-mini",
            "--difficulty",
            "hard",
            "--injection-mode",
            "random",
        ]
    )

    config = _build_config(args)

    assert config.log_directory == "logs/test-run"
    assert config.number_of_tasks == 4
    assert config.effective_sampling_size == 7
    assert config.privileged_llm_model == "openai:gpt-5"
    assert config.privileged_llm_max_attempts == 12
    assert config.quarantined_llm_model == "openai:gpt-4.1"
    assert config.task_generator_model == "openai:gpt-5-mini"
    assert config.evaluator_model == "openai:gpt-4.1-mini"
    assert config.difficulty is TaskGenerationDifficulty.HARD
    assert config.injection_mode is InjectionSelectionMode.RANDOM


def test_qllm_and_evaluator_default_to_pllm_via_none() -> None:
    config = _build_config(_parse_args(["--pllm", "openai:gpt-5"]))

    assert config.quarantined_llm_model is None
    assert config.evaluator_model is None


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--number-of-tasks", "0"),
        ("--sampling-size", "-1"),
        ("--pllm-retry-budget", "0"),
    ),
)
def test_positive_count_options_are_validated(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _parse_args([option, value])
