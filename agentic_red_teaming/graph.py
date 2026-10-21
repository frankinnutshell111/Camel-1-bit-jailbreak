from __future__ import annotations

import enum
import hashlib
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatMessage

from agentic_red_teaming.evaluate_trajectory import evaluate_trajectory
from agentic_red_teaming.log_red_teaming_report import (
    initialize_red_teaming_log,
    log_red_teaming_report,
    log_trajectory_record,
)
from agentic_red_teaming.resources import ResourceLocation, ResourceType
from agentic_red_teaming.run_bstw_trajectory import run_bstw_trajectory
from agentic_red_teaming.select_injection_targets import select_injection_targets
from agentic_red_teaming.task_generator import task_generator_agent


class TaskGenerationDifficulty(str, enum.Enum):
    """Requested difficulty for the generated BSTW user task."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InjectionSelectionMode(str, enum.Enum):
    """How injectable resources should be selected from the ground truth."""

    NONE = "none"
    SINGLE = "single"
    RANDOM = "random"
    RANDOM_SUBSET = "random"
    ALL = "all"


@dataclass(frozen=True)
class RedTeamingConfig:
    """Configuration shared by every stage of one experiment."""

    difficulty: TaskGenerationDifficulty
    injection_mode: InjectionSelectionMode
    num_retries: int | None = None
    number_of_tasks: int = 1
    sampling_size: int = 1
    suite_name: str = "bstw"
    log_directory: str = "logs/agentic_red_teaming"
    random_seed: int | None = None
    task_generator_model: str = "openai:gpt-4.1-2025-04-14"
    task_generator_retries: int = 3
    privileged_llm_model: str = "openai:gpt-4.1-2025-04-14"
    quarantined_llm_model: str | None = None
    evaluator_model: str | None = None
    privileged_llm_max_attempts: int = 10
    quarantined_llm_retries: int = 10
    evaluator_retries: int = 3
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    thinking_budget_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.suite_name != "bstw":
            raise ValueError("The initial framework scaffold supports only the BSTW suite.")
        if self.number_of_tasks < 1:
            raise ValueError("number_of_tasks must be at least 1.")
        if self.sampling_size < 1:
            raise ValueError("sampling_size must be at least 1.")
        if self.num_retries is not None and self.num_retries < 1:
            raise ValueError("num_retries must be at least 1 when provided.")
        if self.num_retries is not None and self.sampling_size != 1 and self.sampling_size != self.num_retries:
            raise ValueError(
                "num_retries is a compatibility alias for sampling_size; do not provide conflicting values."
            )
        if self.task_generator_retries < 0:
            raise ValueError("task_generator_retries cannot be negative.")
        if self.privileged_llm_max_attempts < 1:
            raise ValueError("privileged_llm_max_attempts must be at least 1.")
        if self.quarantined_llm_retries < 0:
            raise ValueError("quarantined_llm_retries cannot be negative.")
        if self.evaluator_retries < 0:
            raise ValueError("evaluator_retries cannot be negative.")
        if self.thinking_budget_tokens is not None and self.thinking_budget_tokens < 1:
            raise ValueError("thinking_budget_tokens must be positive when provided.")

    @property
    def effective_sampling_size(self) -> int:
        """Return the sample count, honoring the former ``num_retries`` name."""

        return self.num_retries if self.num_retries is not None else self.sampling_size


@dataclass(frozen=True)
class GeneratedUserTask:
    """Task-generator output before registration with an AgentDojo suite."""

    task_id: str
    prompt: str
    difficulty: TaskGenerationDifficulty
    ground_truth: tuple[ResourceLocation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectionTarget:
    """One concrete untrusted resource selected for runtime injection."""

    ground_truth_resource_index: int
    resource_type: ResourceType
    resource_id: str
    injection_vector: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedInjectionTask:
    """Runtime injection specification derived from a generated user task."""

    task_id: str
    mode: InjectionSelectionMode
    payload: str | None
    candidates: tuple[InjectionTarget, ...]
    selected_targets: tuple[InjectionTarget, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """One independent execution of the generated task and injection task."""

    trajectory_id: str
    retry_index: int
    messages: tuple[ChatMessage, ...]
    model_output: str
    tool_calls: tuple[FunctionCall, ...]
    task_completed: bool | None
    log_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sample_index(self) -> int:
        """Return the within-task sample index under its clearer new name."""

        return self.retry_index


@dataclass(frozen=True)
class PlanEvaluation:
    """Evaluator judgment for one P-LLM-generated code plan."""

    plan_index: int
    plan: str
    jailbreak: bool
    dramatic_deviation: bool
    unsafe_risk: bool
    goal_drift: bool
    scope_expansion: bool
    unnecessary_privilege_use: bool
    other: bool
    explanation: str

    def __post_init__(self) -> None:
        if self.plan_index < 0:
            raise ValueError("plan_index cannot be negative.")
        if self.jailbreak != (self.dramatic_deviation or self.unsafe_risk):
            raise ValueError("jailbreak must equal dramatic_deviation OR unsafe_risk.")
        labels = (self.goal_drift, self.scope_expansion, self.unnecessary_privilege_use, self.other)
        if self.jailbreak != any(labels):
            raise ValueError("jailbreak must have at least one mechanism label, and safe plans must have none.")


@dataclass(frozen=True)
class TrajectoryEvaluation:
    """Evaluator judgment for one trajectory."""

    trajectory_id: str
    jailbreak: bool
    deviation_score: float
    deviated_from_task: bool
    unsafe_risk: bool
    goal_drift: bool
    scope_expansion: bool
    unnecessary_privilege_use: bool
    other: bool
    rationale: str
    explanations: tuple[str, ...] = ()
    unsafe_events: tuple[str, ...] = ()
    plan_evaluations: tuple[PlanEvaluation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.deviation_score <= 1.0:
            raise ValueError("deviation_score must be between 0.0 and 1.0.")
        if self.jailbreak != (self.deviated_from_task or self.unsafe_risk):
            raise ValueError("jailbreak must equal deviated_from_task OR unsafe_risk.")
        labels = (self.goal_drift, self.scope_expansion, self.unnecessary_privilege_use, self.other)
        if self.jailbreak != any(labels):
            raise ValueError("jailbreak must have at least one mechanism label, and safe trajectories must have none.")


@dataclass(frozen=True)
class RedTeamingReport:
    """Compact aggregate returned after all trajectory records are streamed."""

    report_id: str
    created_at_utc: str
    config: RedTeamingConfig
    number_of_tasks: int
    user_task_ids: tuple[str, ...]
    sampled_trajectory_count: int
    jailbroken_trajectory_count: int
    jailbroken_trajectory_ids: tuple[str, ...]
    goal_drift_trajectory_count: int
    scope_expansion_trajectory_count: int
    unnecessary_privilege_use_trajectory_count: int
    other_trajectory_count: int
    log_directory: str
    trajectory_log_directory: str
    report_path: str

    def __post_init__(self) -> None:
        if self.number_of_tasks != self.config.number_of_tasks:
            raise ValueError("number_of_tasks does not match the configuration.")
        if len(self.user_task_ids) != self.number_of_tasks:
            raise ValueError("user_task_ids does not match number_of_tasks.")
        expected_trajectory_count = self.number_of_tasks * self.config.effective_sampling_size
        if self.sampled_trajectory_count != expected_trajectory_count:
            raise ValueError("sampled_trajectory_count must equal number_of_tasks * sampling_size.")
        if self.jailbroken_trajectory_count != len(self.jailbroken_trajectory_ids):
            raise ValueError("jailbroken_trajectory_count does not match jailbroken_trajectory_ids.")
        if len(set(self.jailbroken_trajectory_ids)) != len(self.jailbroken_trajectory_ids):
            raise ValueError("jailbroken_trajectory_ids must be unique.")
        if self.jailbroken_trajectory_count > self.sampled_trajectory_count:
            raise ValueError("jailbroken_trajectory_count cannot exceed sampled_trajectory_count.")
        label_counts = self.jailbreak_label_counts.values()
        if any(count < 0 for count in label_counts):
            raise ValueError("jailbreak label counts cannot be negative.")
        if any(count > self.jailbroken_trajectory_count for count in label_counts):
            raise ValueError("jailbreak label counts cannot exceed jailbroken_trajectory_count.")
        if sum(label_counts) < self.jailbroken_trajectory_count:
            raise ValueError("every jailbroken trajectory must have at least one mechanism label.")

    @property
    def unsafe_trajectory_count(self) -> int:
        """Compatibility alias for the aggregate jailbreak count."""

        return self.jailbroken_trajectory_count

    @property
    def jailbreak_label_counts(self) -> dict[str, int]:
        """Return overlapping trajectory counts for the jailbreak taxonomy."""

        return {
            "goal_drift": self.goal_drift_trajectory_count,
            "scope_expansion": self.scope_expansion_trajectory_count,
            "unnecessary_privilege_use": self.unnecessary_privilege_use_trajectory_count,
            "other": self.other_trajectory_count,
        }


def _sample_selection_config(
    config: RedTeamingConfig,
    task_index: int,
    sample_index: int,
) -> RedTeamingConfig:
    """Derive a reproducible injection-selection seed for one sample."""

    if config.random_seed is None:
        return config
    seed_material = f"{config.random_seed}:task:{task_index}:sample:{sample_index}"
    sample_seed = int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    return replace(config, random_seed=sample_seed)


def _new_report_identity() -> tuple[str, str]:
    """Return a collision-resistant run ID and its UTC creation timestamp."""

    created_at = datetime.now(timezone.utc)
    report_id = f"run_{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    return report_id, created_at.isoformat()


def run_red_teaming_graph(config: RedTeamingConfig) -> RedTeamingReport:
    """Run the complete red-teaming workflow in stage order."""

    user_tasks: list[GeneratedUserTask] = []
    previous_queries: list[str] = []
    for task_index in range(config.number_of_tasks):
        generated_task = task_generator_agent(
            config,
            diversity_index=task_index,
            previous_queries=previous_queries,
        )
        user_task = replace(
            generated_task,
            metadata={**generated_task.metadata, "batch_index": task_index},
        )
        user_tasks.append(user_task)
        previous_queries.append(user_task.prompt)

    report_id, created_at_utc = _new_report_identity()
    log_session = initialize_red_teaming_log(
        report_id,
        config.log_directory,
    )
    sampled_trajectory_count = 0
    jailbroken_trajectory_ids: list[str] = []
    goal_drift_trajectory_count = 0
    scope_expansion_trajectory_count = 0
    unnecessary_privilege_use_trajectory_count = 0
    other_trajectory_count = 0
    for task_index, user_task in enumerate(user_tasks):
        for sample_index in range(config.effective_sampling_size):
            try:
                selection_config = _sample_selection_config(
                    config,
                    task_index,
                    sample_index,
                )
                injection_task = select_injection_targets(
                    user_task.ground_truth,
                    selection_config,
                )
                trajectory = run_bstw_trajectory(
                    user_task,
                    injection_task,
                    sample_index,
                    config,
                )
                trajectory = log_trajectory_record(
                    log_session,
                    sampled_trajectory_count,
                    user_task,
                    injection_task,
                    trajectory,
                )
                evaluation = evaluate_trajectory(
                    trajectory,
                    user_task.prompt,
                    config,
                )
                log_trajectory_record(
                    log_session,
                    sampled_trajectory_count,
                    user_task,
                    injection_task,
                    trajectory,
                    evaluation,
                )
                if evaluation.jailbreak:
                    jailbroken_trajectory_ids.append(trajectory.trajectory_id)
                if evaluation.goal_drift:
                    goal_drift_trajectory_count += 1
                if evaluation.scope_expansion:
                    scope_expansion_trajectory_count += 1
                if evaluation.unnecessary_privilege_use:
                    unnecessary_privilege_use_trajectory_count += 1
                if evaluation.other:
                    other_trajectory_count += 1
            except Exception as error:
                print(
                    "Skipping trajectory attempt "
                    f"task_index={task_index} sample_index={sample_index} after "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                # Count every requested sample as attempted so the compact report
                # remains consistent even when one trajectory is skipped.
                sampled_trajectory_count += 1

    report = RedTeamingReport(
        report_id=report_id,
        created_at_utc=created_at_utc,
        config=config,
        number_of_tasks=len(user_tasks),
        user_task_ids=tuple(task.task_id for task in user_tasks),
        sampled_trajectory_count=sampled_trajectory_count,
        jailbroken_trajectory_count=len(jailbroken_trajectory_ids),
        jailbroken_trajectory_ids=tuple(jailbroken_trajectory_ids),
        goal_drift_trajectory_count=goal_drift_trajectory_count,
        scope_expansion_trajectory_count=scope_expansion_trajectory_count,
        unnecessary_privilege_use_trajectory_count=unnecessary_privilege_use_trajectory_count,
        other_trajectory_count=other_trajectory_count,
        log_directory=str(log_session.run_directory),
        trajectory_log_directory=str(log_session.trajectory_directory),
        report_path=str(log_session.report_path),
    )
    return log_red_teaming_report(report)


__all__ = [
    "GeneratedInjectionTask",
    "GeneratedUserTask",
    "InjectionSelectionMode",
    "InjectionTarget",
    "PlanEvaluation",
    "RedTeamingConfig",
    "RedTeamingReport",
    "ResourceLocation",
    "ResourceType",
    "TaskGenerationDifficulty",
    "Trajectory",
    "TrajectoryEvaluation",
    "evaluate_trajectory",
    "log_red_teaming_report",
    "run_bstw_trajectory",
    "run_red_teaming_graph",
    "select_injection_targets",
    "task_generator_agent",
]
