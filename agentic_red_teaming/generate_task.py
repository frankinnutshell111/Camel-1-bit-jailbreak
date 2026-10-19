"""Generate and print one BSTW red-teaming user task."""

from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import replace
from typing import Any

from agentic_red_teaming.graph import (
    GeneratedUserTask,
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
)
from agentic_red_teaming.task_generator import task_generator_agent


def _task_to_dict(task: GeneratedUserTask) -> dict[str, Any]:
    """Convert a generated task and its Pydantic locations to JSON data."""

    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "difficulty": task.difficulty.value,
        "ground_truth": [location.model_dump(mode="json") for location in task.ground_truth],
        "metadata": dict(task.metadata),
    }


def main() -> None:
    """Run the task generator once and print its structured result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "difficulty",
        choices=[difficulty.value for difficulty in TaskGenerationDifficulty],
        help="Difficulty of the task to generate.",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-5",
        help="PydanticAI model identifier (default: openai:gpt-5).",
    )
    parser.add_argument(
        "--model-retries",
        type=int,
        default=2,
        help="Structured-output retry budget (default: 2).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of diverse tasks to generate in one batch (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible diversity assignments.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    config = RedTeamingConfig(
        difficulty=TaskGenerationDifficulty(args.difficulty),
        injection_mode=InjectionSelectionMode.NONE,
        sampling_size=1,
        task_generator_model=args.model,
        task_generator_retries=args.model_retries,
        random_seed=args.seed,
    )
    batch_seed = args.seed if args.seed is not None else secrets.randbits(64)
    config = replace(config, random_seed=batch_seed)

    tasks: list[GeneratedUserTask] = []
    previous_queries: list[str] = []
    for diversity_index in range(args.count):
        task = task_generator_agent(
            config,
            diversity_index=diversity_index,
            previous_queries=previous_queries,
        )
        tasks.append(task)
        previous_queries.append(task.prompt)

    output: dict[str, Any]
    if args.count == 1:
        output = _task_to_dict(tasks[0])
    else:
        output = {
            "seed": batch_seed,
            "tasks": [_task_to_dict(task) for task in tasks],
        }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
