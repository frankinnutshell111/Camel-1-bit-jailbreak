"""Streaming JSON logging for agentic red-teaming experiments."""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentic_red_teaming.graph import (
        GeneratedInjectionTask,
        GeneratedUserTask,
        RedTeamingReport,
        Trajectory,
        TrajectoryEvaluation,
    )


@dataclass(frozen=True)
class RedTeamingLogSession:
    """Filesystem locations allocated for one graph run."""

    report_id: str
    run_directory: Path
    trajectory_directory: Path
    report_path: Path


def _to_jsonable(value: Any) -> Any:
    """Recursively convert framework, Pydantic, and AgentDojo values to JSON."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, enum.Enum):
        return _to_jsonable(value.value)
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_jsonable(item) for item in value]
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON document atomically."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(_to_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def initialize_red_teaming_log(
    report_id: str,
    log_directory: str,
) -> RedTeamingLogSession:
    """Create the run directory before any trajectories are sampled."""

    log_root = Path(log_directory).expanduser()
    if not log_root.is_absolute():
        log_root = Path.cwd() / log_root
    run_directory = (log_root / report_id).resolve()
    trajectory_directory = run_directory / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=False)
    return RedTeamingLogSession(
        report_id=report_id,
        run_directory=run_directory,
        trajectory_directory=trajectory_directory,
        report_path=run_directory / "report.json",
    )


def log_trajectory_record(
    session: RedTeamingLogSession,
    record_index: int,
    user_task: GeneratedUserTask,
    injection_task: GeneratedInjectionTask,
    trajectory: Trajectory,
    evaluation: TrajectoryEvaluation | None = None,
) -> Trajectory:
    """Write or finalize one trajectory file and return it with its log path.

    The graph first calls this immediately after CaMeL returns, leaving the
    evaluation field null. It calls the function again after evaluation to
    atomically replace the same file with the complete record.
    """

    if record_index < 0:
        raise ValueError("record_index cannot be negative.")
    trajectory_path = session.trajectory_directory / (f"{record_index:05d}_{trajectory.trajectory_id}.json")
    logged_trajectory = replace(
        trajectory,
        log_path=str(trajectory_path.resolve()),
    )
    _write_json(
        trajectory_path,
        {
            "report_id": session.report_id,
            "status": "complete" if evaluation is not None else "awaiting_evaluation",
            "user_task": user_task,
            "injection_task": injection_task,
            "trajectory": logged_trajectory,
            "evaluation": evaluation,
        },
    )
    return logged_trajectory


def log_red_teaming_report(report: RedTeamingReport) -> RedTeamingReport:
    """Write the compact final report without retaining trajectory contents."""

    report_path = Path(report.report_path)
    _write_json(
        report_path,
        {
            "report_id": report.report_id,
            "created_at_utc": report.created_at_utc,
            "config": report.config,
            "number_of_tasks": report.number_of_tasks,
            "sampling_size": report.config.effective_sampling_size,
            "sampled_trajectory_count": report.sampled_trajectory_count,
            "jailbroken_trajectory_count": report.jailbroken_trajectory_count,
            "jailbroken_trajectory_ids": report.jailbroken_trajectory_ids,
            "user_task_ids": report.user_task_ids,
            "trajectory_log_directory": report.trajectory_log_directory,
        },
    )
    return report


__all__ = [
    "RedTeamingLogSession",
    "initialize_red_teaming_log",
    "log_red_teaming_report",
    "log_trajectory_record",
]
