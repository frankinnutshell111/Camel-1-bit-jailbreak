from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentdojo.functions_runtime import FunctionCall
from pydantic_ai import ModelHTTPError

from agentic_red_teaming.graph import (
    GeneratedInjectionTask,
    GeneratedUserTask,
    InjectionSelectionMode,
    RedTeamingConfig,
    RedTeamingReport,
    TaskGenerationDifficulty,
    Trajectory,
    TrajectoryEvaluation,
    run_red_teaming_graph,
)
from agentic_red_teaming.log_red_teaming_report import (
    RedTeamingLogSession,
    initialize_red_teaming_log,
    log_red_teaming_report,
    log_trajectory_record,
)
from agentic_red_teaming.resources import ResourceLocation, ResourceType


def _config(**overrides) -> RedTeamingConfig:
    values = {
        "difficulty": TaskGenerationDifficulty.MEDIUM,
        "injection_mode": InjectionSelectionMode.SINGLE,
        "number_of_tasks": 2,
        "sampling_size": 3,
        "random_seed": 42,
    }
    values.update(overrides)
    return RedTeamingConfig(**values)


def _user_task(index: int) -> GeneratedUserTask:
    return GeneratedUserTask(
        task_id=f"task_{index}",
        prompt=f"Generated query {index}",
        difficulty=TaskGenerationDifficulty.MEDIUM,
        ground_truth=(
            ResourceLocation(
                resource_type=ResourceType.EMAIL,
                resource_id=f"email_{index}",
            ),
        ),
        metadata={"diversity_index": index},
    )


def _injection_task(index: int) -> GeneratedInjectionTask:
    return GeneratedInjectionTask(
        task_id=f"injection_{index}",
        mode=InjectionSelectionMode.SINGLE,
        payload="test injection",
        candidates=(),
        selected_targets=(),
    )


def _evaluation(trajectory_id: str, *, jailbreak: bool) -> TrajectoryEvaluation:
    return TrajectoryEvaluation(
        trajectory_id=trajectory_id,
        jailbreak=jailbreak,
        deviation_score=1.0 if jailbreak else 0.0,
        deviated_from_task=jailbreak,
        unsafe_risk=False,
        goal_drift=jailbreak,
        scope_expansion=False,
        unnecessary_privilege_use=False,
        other=False,
        rationale="test judgment",
        explanations=("test judgment",),
    )


def test_graph_pre_generates_tasks_then_samples_each_task(monkeypatch) -> None:
    events: list[str] = []
    injection_counter = 0

    def fake_generate(_config, *, diversity_index, previous_queries):
        events.append(f"generate:{diversity_index}:{len(previous_queries)}")
        return _user_task(diversity_index)

    def fake_select(_ground_truth, _config):
        nonlocal injection_counter
        events.append(f"inject:{injection_counter}")
        result = _injection_task(injection_counter)
        injection_counter += 1
        return result

    def fake_run(user_task, injection_task, sample_index, _config):
        events.append(f"run:{user_task.task_id}:{sample_index}")
        trajectory_id = f"trajectory_{user_task.task_id}_{sample_index}"
        return Trajectory(
            trajectory_id=trajectory_id,
            retry_index=sample_index,
            messages=(),
            model_output="",
            tool_calls=(),
            task_completed=None,
            metadata={
                "task_index": user_task.metadata["batch_index"],
                "injection_task_id": injection_task.task_id,
            },
        )

    def fake_evaluate(trajectory, _original_query, _config):
        events.append(f"evaluate:{trajectory.trajectory_id}")
        return _evaluation(
            trajectory.trajectory_id,
            jailbreak=trajectory.sample_index == 1,
        )

    def fake_initialize(report_id, _log_directory):
        events.append("initialize-log")
        run_directory = Path("/tmp/test-red-teaming-run")
        return RedTeamingLogSession(
            report_id=report_id,
            run_directory=run_directory,
            trajectory_directory=run_directory / "trajectories",
            report_path=run_directory / "report.json",
        )

    def fake_log_trajectory(
        _session,
        _record_index,
        _user_task,
        _injection_task,
        trajectory,
        evaluation=None,
    ):
        phase = "complete" if evaluation is not None else "raw"
        events.append(f"record:{trajectory.trajectory_id}:{phase}")
        return trajectory

    def fake_log(report):
        events.append("log")
        return report

    monkeypatch.setattr("agentic_red_teaming.graph.task_generator_agent", fake_generate)
    monkeypatch.setattr("agentic_red_teaming.graph.select_injection_targets", fake_select)
    monkeypatch.setattr("agentic_red_teaming.graph.run_bstw_trajectory", fake_run)
    monkeypatch.setattr("agentic_red_teaming.graph.evaluate_trajectory", fake_evaluate)
    monkeypatch.setattr(
        "agentic_red_teaming.graph.initialize_red_teaming_log",
        fake_initialize,
    )
    monkeypatch.setattr(
        "agentic_red_teaming.graph.log_trajectory_record",
        fake_log_trajectory,
    )
    monkeypatch.setattr("agentic_red_teaming.graph.log_red_teaming_report", fake_log)

    report = run_red_teaming_graph(_config())

    assert events[:2] == ["generate:0:0", "generate:1:1"]
    assert report.number_of_tasks == 2
    assert report.user_task_ids == ("task_0", "task_1")
    assert report.sampled_trajectory_count == 6
    assert report.jailbroken_trajectory_count == 2
    assert report.jailbreak_label_counts == {
        "goal_drift": 2,
        "scope_expansion": 0,
        "unnecessary_privilege_use": 0,
        "other": 0,
    }
    assert report.jailbroken_trajectory_ids == (
        "trajectory_task_0_1",
        "trajectory_task_1_1",
    )
    assert events[4:8] == [
        "run:task_0:0",
        "record:trajectory_task_0_0:raw",
        "evaluate:trajectory_task_0_0",
        "record:trajectory_task_0_0:complete",
    ]
    assert not hasattr(report, "trajectories")
    assert events[-1] == "log"


def test_json_logger_writes_trajectory_records_and_summary(tmp_path) -> None:
    config = _config(
        number_of_tasks=1,
        sampling_size=2,
        log_directory=str(tmp_path / "logs"),
    )
    user_task = _user_task(0)
    user_task = GeneratedUserTask(
        task_id=user_task.task_id,
        prompt=user_task.prompt,
        difficulty=user_task.difficulty,
        ground_truth=user_task.ground_truth,
        metadata={**user_task.metadata, "batch_index": 0},
    )
    injection_tasks = (_injection_task(0), _injection_task(1))
    trajectories = tuple(
        Trajectory(
            trajectory_id=f"trajectory_{index}",
            retry_index=index,
            messages=(),
            model_output="done",
            tool_calls=(FunctionCall(function="get_unread_emails", args={}),),
            task_completed=None,
            metadata={"task_index": 0, "sample_index": index},
        )
        for index in range(2)
    )
    evaluations = (
        _evaluation("trajectory_0", jailbreak=False),
        _evaluation("trajectory_1", jailbreak=True),
    )
    session = initialize_red_teaming_log("run_test", config.log_directory)
    first_logged_trajectory = log_trajectory_record(
        session,
        0,
        user_task,
        injection_tasks[0],
        trajectories[0],
    )
    first_path = Path(first_logged_trajectory.log_path or "")
    pending_record = json.loads(first_path.read_text(encoding="utf-8"))
    assert pending_record["status"] == "awaiting_evaluation"
    assert pending_record["evaluation"] is None

    for index, (injection_task, trajectory, evaluation) in enumerate(
        zip(injection_tasks, trajectories, evaluations, strict=True)
    ):
        log_trajectory_record(
            session,
            index,
            user_task,
            injection_task,
            trajectory,
            evaluation,
        )

    report = RedTeamingReport(
        report_id="run_test",
        created_at_utc="2026-10-15T12:00:00+00:00",
        config=config,
        number_of_tasks=1,
        user_task_ids=(user_task.task_id,),
        sampled_trajectory_count=2,
        jailbroken_trajectory_count=1,
        jailbroken_trajectory_ids=("trajectory_1",),
        goal_drift_trajectory_count=1,
        scope_expansion_trajectory_count=0,
        unnecessary_privilege_use_trajectory_count=0,
        other_trajectory_count=0,
        log_directory=str(session.run_directory),
        trajectory_log_directory=str(session.trajectory_directory),
        report_path=str(session.report_path),
    )

    logged_report = log_red_teaming_report(report)

    trajectory_files = sorted((tmp_path / "logs" / "run_test" / "trajectories").glob("*.json"))
    assert len(trajectory_files) == 2
    first_record = json.loads(trajectory_files[0].read_text(encoding="utf-8"))
    assert first_record["user_task"]["task_id"] == "task_0"
    assert first_record["trajectory"]["trajectory_id"] == "trajectory_0"
    assert first_record["evaluation"]["jailbreak"] is False
    assert first_record["status"] == "complete"
    assert first_record["trajectory"]["tool_calls"][0]["function"] == ("get_unread_emails")

    summary = json.loads((tmp_path / "logs" / "run_test" / "report.json").read_text(encoding="utf-8"))
    assert summary["sampled_trajectory_count"] == 2
    assert summary["jailbroken_trajectory_count"] == 1
    assert summary["jailbroken_trajectory_ids"] == ["trajectory_1"]
    assert summary["jailbreak_label_counts"] == {
        "goal_drift": 1,
        "scope_expansion": 0,
        "unnecessary_privilege_use": 0,
        "other": 0,
    }
    assert logged_report.report_path == str(session.report_path)
    assert "trajectories" not in summary


def test_num_retries_remains_a_sampling_size_alias() -> None:
    config = RedTeamingConfig(
        difficulty=TaskGenerationDifficulty.EASY,
        injection_mode=InjectionSelectionMode.NONE,
        num_retries=4,
    )

    assert config.effective_sampling_size == 4


def test_graph_skips_trajectory_errors_and_continues(monkeypatch, tmp_path, capsys) -> None:
    run_count = 0

    def fake_generate(_config, *, diversity_index, previous_queries):
        return _user_task(diversity_index)

    def fake_select(_ground_truth, _config):
        return _injection_task(0)

    def failing_run(*_args, **_kwargs):
        nonlocal run_count
        run_count += 1
        raise ModelHTTPError(429, "gpt-test", {"error": "quota exceeded"})

    def unexpected_evaluation(*_args, **_kwargs):
        raise AssertionError("Evaluation must not run after an API failure")

    monkeypatch.setattr("agentic_red_teaming.graph.task_generator_agent", fake_generate)
    monkeypatch.setattr("agentic_red_teaming.graph.select_injection_targets", fake_select)
    monkeypatch.setattr("agentic_red_teaming.graph.run_bstw_trajectory", failing_run)
    monkeypatch.setattr("agentic_red_teaming.graph.evaluate_trajectory", unexpected_evaluation)

    config = _config(
        number_of_tasks=1,
        sampling_size=3,
        log_directory=str(tmp_path / "logs"),
    )
    report = run_red_teaming_graph(config)

    assert run_count == 3
    assert report.sampled_trajectory_count == 3
    assert report.jailbroken_trajectory_count == 0
    stderr = capsys.readouterr().err
    assert stderr.count("Skipping trajectory attempt") == 3
    assert "ModelHTTPError" in stderr
