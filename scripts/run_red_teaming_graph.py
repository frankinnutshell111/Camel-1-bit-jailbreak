#!/usr/bin/env python3
"""Run the complete agentic red-teaming graph from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agentic_red_teaming.graph import (
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
    run_red_teaming_graph,
)

DEFAULT_MODEL = "openai:gpt-4.1-2025-04-14"
DEFAULT_LOG_FOLDER = "logs/agentic_red_teaming"
DEFAULT_PLLM_MAX_ATTEMPTS = 10


def _positive_integer(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _model_identifier(value: str) -> str:
    """Validate the provider:model convention used by PydanticAI."""

    provider, separator, model = value.partition(":")
    if not separator or not provider.strip() or not model.strip():
        raise argparse.ArgumentTypeError("model must use provider:model format, for example openai:gpt-5")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Construct the graph runner's command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-folder",
        default=DEFAULT_LOG_FOLDER,
        help=f"Root folder for run logs (default: {DEFAULT_LOG_FOLDER}).",
    )
    parser.add_argument(
        "--number-of-tasks",
        type=_positive_integer,
        default=1,
        help="Number of user tasks to generate before sampling (default: 1).",
    )
    parser.add_argument(
        "--sampling-size",
        type=_positive_integer,
        default=1,
        help="Independent trajectories sampled per generated task (default: 1).",
    )
    parser.add_argument(
        "--pllm",
        "--p-llm",
        dest="pllm",
        type=_model_identifier,
        default=DEFAULT_MODEL,
        help=f"Privileged LLM in provider:model format (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--pllm-retry-budget",
        "--pllm-max-attempts",
        dest="pllm_max_attempts",
        type=_positive_integer,
        default=DEFAULT_PLLM_MAX_ATTEMPTS,
        help=(
            "Maximum total P-LLM plan attempts per trajectory, including the "
            f"initial plan (default: {DEFAULT_PLLM_MAX_ATTEMPTS})."
        ),
    )
    parser.add_argument(
        "--qllm",
        "--q-llm",
        dest="qllm",
        type=_model_identifier,
        help="Quarantined LLM; defaults to the selected P-LLM.",
    )
    parser.add_argument(
        "--task-generator-llm",
        type=_model_identifier,
        default=DEFAULT_MODEL,
        help=f"Task-generator LLM (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--evaluation-llm",
        type=_model_identifier,
        help="Trajectory-evaluator LLM; defaults to the selected P-LLM.",
    )
    parser.add_argument(
        "--difficulty",
        choices=[difficulty.value for difficulty in TaskGenerationDifficulty],
        default=TaskGenerationDifficulty.MEDIUM.value,
        help="Generated task difficulty (default: medium).",
    )
    parser.add_argument(
        "--injection-mode",
        choices=[
            InjectionSelectionMode.NONE.value,
            InjectionSelectionMode.SINGLE.value,
            InjectionSelectionMode.RANDOM.value,
            InjectionSelectionMode.ALL.value,
        ],
        default=InjectionSelectionMode.SINGLE.value,
        help="Runtime injection-target selection mode (default: single).",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, accepting an explicit sequence for tests."""

    return _build_parser().parse_args(argv)


def _build_config(args: argparse.Namespace) -> RedTeamingConfig:
    """Translate parsed CLI values into the graph's configuration."""

    return RedTeamingConfig(
        difficulty=TaskGenerationDifficulty(args.difficulty),
        injection_mode=InjectionSelectionMode(args.injection_mode),
        number_of_tasks=args.number_of_tasks,
        sampling_size=args.sampling_size,
        log_directory=args.log_folder,
        privileged_llm_model=args.pllm,
        privileged_llm_max_attempts=args.pllm_max_attempts,
        quarantined_llm_model=args.qllm,
        task_generator_model=args.task_generator_llm,
        evaluator_model=args.evaluation_llm,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the graph and print the location and aggregate result."""

    config = _build_config(_parse_args(argv))
    total_trajectories = config.number_of_tasks * config.effective_sampling_size
    print(
        f"Generating {config.number_of_tasks} task(s) and sampling "
        f"{total_trajectories} total trajectory/trajectories...",
        flush=True,
    )

    report = run_red_teaming_graph(config)
    print(
        json.dumps(
            {
                "report_id": report.report_id,
                "sampled_trajectory_count": report.sampled_trajectory_count,
                "jailbroken_trajectory_count": (report.jailbroken_trajectory_count),
                "jailbroken_trajectory_ids": report.jailbroken_trajectory_ids,
                "jailbreak_label_counts": report.jailbreak_label_counts,
                "log_directory": report.log_directory,
                "report_path": report.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
