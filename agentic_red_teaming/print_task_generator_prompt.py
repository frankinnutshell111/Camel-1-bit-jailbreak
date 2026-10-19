"""Print the exact task-generator inputs without making an LLM request."""

from __future__ import annotations

import argparse
import json

from agentic_red_teaming.graph import (
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
)
from agentic_red_teaming.task_generator import build_task_generator_request_preview


def main() -> None:
    """Print the system message, user message, and structured-output schema."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "difficulty",
        choices=[difficulty.value for difficulty in TaskGenerationDifficulty],
        help="Difficulty requested from the task generator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for a reproducible diversity assignment.",
    )
    args = parser.parse_args()

    config = RedTeamingConfig(
        difficulty=TaskGenerationDifficulty(args.difficulty),
        injection_mode=InjectionSelectionMode.NONE,
        sampling_size=1,
        random_seed=args.seed,
    )
    preview = build_task_generator_request_preview(config)
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
