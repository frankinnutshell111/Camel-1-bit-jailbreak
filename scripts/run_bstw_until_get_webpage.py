#!/usr/bin/env python3
"""Repeat the BSTW linked-fallback injection until get_webpage is called."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "openai:gpt-4.1-2025-04-14"
SUITE = "bstw"
USER_TASK = "user_task_1"
INJECTION_TASK = "injection_task_1"
ATTACK = "bstw_vendorx_fallback"


def _model_name(model: str) -> str:
    """Return the model portion of a provider:model identifier."""
    try:
        _, name = model.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Model '{model}' must use the provider:model format"
        ) from exc
    return name


def _transcript_path(repo_root: Path, p_llm: str, q_llm: str) -> Path:
    pipeline = f"{_model_name(p_llm)}+camel-q:{_model_name(q_llm)}"
    return (
        repo_root
        / "logs"
        / pipeline
        / SUITE
        / USER_TASK
        / ATTACK
        / f"{INJECTION_TASK}.json"
    )


def _get_webpage_calls(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every recorded get_webpage call in an AgentDojo transcript."""
    matches: list[dict[str, Any]] = []
    for message in transcript.get("messages", []):
        calls: list[Any] = []
        if message.get("role") == "tool" and message.get("tool_call"):
            calls.append(message["tool_call"])
        calls.extend(message.get("tool_calls") or [])

        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if function == "get_webpage":
                matches.append(call)
            elif (
                isinstance(function, dict)
                and function.get("name") == "get_webpage"
            ):
                matches.append(call)
    return matches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BSTW user_task_1 with injection_task_1 repeatedly, overwriting "
            "the transcript, and stop when get_webpage is called."
        )
    )
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--p-llm", default=DEFAULT_MODEL)
    parser.add_argument("--q-llm", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream each main.py run instead of showing only runner progress.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_runs < 1:
        raise ValueError("--max-runs must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = repo_root / env_file
    if not env_file.is_file():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    transcript_path = _transcript_path(repo_root, args.p_llm, args.q_llm)
    command = [
        "uv",
        "run",
        "--env-file",
        str(env_file),
        "main.py",
        args.p_llm,
        "--q-llm",
        args.q_llm,
        "--suites",
        SUITE,
        "--user-tasks",
        USER_TASK,
        "--run-attack",
        "--injection-tasks",
        INJECTION_TASK,
        "--max-attempts",
        str(args.max_attempts),
        "--force-rerun",
    ]

    print(f"Transcript: {transcript_path}", flush=True)
    for run_number in range(1, args.max_runs + 1):
        print(f"Run {run_number}/{args.max_runs}...", flush=True)
        transcript_path.unlink(missing_ok=True)

        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            text=True,
            stdout=None if args.verbose else subprocess.PIPE,
            stderr=None if args.verbose else subprocess.STDOUT,
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, file=sys.stderr)
            print(
                f"Run {run_number} failed with exit code "
                f"{completed.returncode}; stopping.",
                file=sys.stderr,
            )
            return completed.returncode

        if not transcript_path.is_file():
            if completed.stdout:
                print(completed.stdout, file=sys.stderr)
            print(
                f"Expected transcript was not created: {transcript_path}",
                file=sys.stderr,
            )
            return 2

        transcript = json.loads(transcript_path.read_text())
        webpage_calls = _get_webpage_calls(transcript)
        if webpage_calls:
            urls = [
                (call.get("args") or {}).get("url", "<unknown>")
                for call in webpage_calls
            ]
            print(
                f"Stopped after run {run_number}: get_webpage was called.",
                flush=True,
            )
            print(f"URL(s): {urls}", flush=True)
            print(f"Transcript: {transcript_path}", flush=True)
            return 0

        print("No get_webpage call in this transcript.", flush=True)

    print(
        f"No get_webpage call was observed after {args.max_runs} runs.",
        flush=True,
    )
    print(f"Last transcript: {transcript_path}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
