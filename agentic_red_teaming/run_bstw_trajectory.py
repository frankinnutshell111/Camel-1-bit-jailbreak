"""Run one generated user task in a freshly injected BSTW environment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str
from pydantic_ai.models import KnownModelName

from agentic_red_teaming.resources import ResourceType

if TYPE_CHECKING:
    from agentic_red_teaming.graph import (
        GeneratedInjectionTask,
        GeneratedUserTask,
        RedTeamingConfig,
        Trajectory,
    )
    from custom_suites.bstw.environment import BSTWEnvironment


def _append_injection(original: str, payload: str) -> str:
    """Append an injection while preserving all legitimate resource content."""

    if not original:
        return payload.strip()
    return f"{original.rstrip()}\n\n{payload.strip()}"


def _apply_slack_injection(
    environment: BSTWEnvironment,
    resource_id: str,
    payload: str,
) -> None:
    """Inject a Slack message addressed by manifest mailbox, destination, and index."""

    try:
        mailbox_name, destination, raw_index = resource_id.split(":", maxsplit=2)
        message_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"Invalid Slack resource ID: {resource_id!r}") from exc

    if mailbox_name not in {"user_inbox", "channel_inbox"}:
        raise ValueError(f"Unknown Slack mailbox in resource ID: {resource_id!r}")
    mailbox = getattr(environment.slack, mailbox_name)
    messages = mailbox.get(destination)
    if messages is None or not 0 <= message_index < len(messages):
        raise ValueError(f"Slack resource not found: {resource_id!r}")
    messages[message_index].body = _append_injection(
        messages[message_index].body,
        payload,
    )


def apply_injection_plan(
    environment: BSTWEnvironment,
    injection_task: GeneratedInjectionTask,
) -> BSTWEnvironment:
    """Apply an injection plan to a deep copy of a clean BSTW environment."""

    attacked_environment = environment.model_copy(deep=True)
    if not injection_task.selected_targets:
        return attacked_environment
    if injection_task.payload is None:
        raise ValueError("An injection plan with selected targets must contain a payload.")

    payload = injection_task.payload
    for target in injection_task.selected_targets:
        resource_id = target.resource_id
        if target.resource_type is ResourceType.EMAIL:
            email = attacked_environment.inbox.emails.get(resource_id)
            if email is None:
                raise ValueError(f"Email resource not found: {resource_id!r}")
            email.body = _append_injection(email.body, payload)
        elif target.resource_type is ResourceType.CALENDAR_EVENT:
            event = attacked_environment.calendar.events.get(resource_id)
            if event is None:
                raise ValueError(f"Calendar resource not found: {resource_id!r}")
            event.description = _append_injection(event.description, payload)
        elif target.resource_type is ResourceType.CLOUD_FILE:
            cloud_file = attacked_environment.cloud_drive.files.get(resource_id)
            if cloud_file is None:
                raise ValueError(f"Cloud-file resource not found: {resource_id!r}")
            cloud_file.content = _append_injection(cloud_file.content, payload)
            cloud_file.size = len(cloud_file.content)
        elif target.resource_type is ResourceType.SLACK_MESSAGE:
            _apply_slack_injection(attacked_environment, resource_id, payload)
        elif target.resource_type is ResourceType.WEBPAGE:
            webpage = attacked_environment.web.web_content.get(resource_id)
            if webpage is None:
                raise ValueError(f"Webpage resource not found: {resource_id!r}")
            attacked_environment.web.web_content[resource_id] = _append_injection(
                webpage,
                payload,
            )
        elif target.resource_type is ResourceType.LOCAL_FILE:
            local_file = attacked_environment.filesystem.files.get(resource_id)
            if local_file is None:
                raise ValueError(f"Local-file resource not found: {resource_id!r}")
            attacked_environment.filesystem.files[resource_id] = _append_injection(
                local_file,
                payload,
            )
        else:
            raise ValueError(f"Unsupported injectable resource type: {target.resource_type!r}")

    return attacked_environment


def run_camel_query(
    query: str,
    attacked_environment: BSTWEnvironment,
    config: RedTeamingConfig,
) -> tuple[ChatMessage, ...]:
    """Run CaMeL on a query and supplied BSTW environment and return its transcript."""

    from camel.interpreter.interpreter import MetadataEvalMode
    from camel.models import make_tools_pipeline
    from custom_suites.bstw.task_suite import task_suite

    privileged_model = cast(KnownModelName, config.privileged_llm_model)
    quarantined_model = (
        cast(KnownModelName, config.quarantined_llm_model) if config.quarantined_llm_model is not None else None
    )
    pipeline = make_tools_pipeline(
        model=privileged_model,
        use_original=False,
        replay_with_policies=False,
        attack_name="agentic_red_teaming",
        reasoning_effort=config.reasoning_effort,
        thinking_budget_tokens=config.thinking_budget_tokens,
        suite="bstw",
        ad_defense=None,
        eval_mode=MetadataEvalMode.NORMAL,
        q_llm=quarantined_model,
        max_attempts=config.privileged_llm_max_attempts,
        quarantined_llm_retries=config.quarantined_llm_retries,
    )
    runtime = FunctionsRuntime(task_suite.tools)
    try:
        _, _, _, messages, _ = pipeline.query(
            query,
            runtime,
            attacked_environment,
            messages=[],
            extra_args={},
        )
    except AbortAgentError as exc:
        messages = exc.messages
    return tuple(messages)


def _model_output_from_messages(messages: Sequence[ChatMessage]) -> str:
    """Return the latest non-empty assistant text in the transcript."""

    for message in reversed(messages):
        if message["role"] != "assistant" or message["content"] is None:
            continue
        return get_text_content_as_str(message["content"])
    return ""


def _tool_calls_from_messages(
    messages: Sequence[ChatMessage],
) -> tuple[FunctionCall, ...]:
    """Collect unique tool calls, including calls made before failed CaMeL plans."""

    tool_calls: list[FunctionCall] = []
    seen_call_objects: set[int] = set()
    for message in messages:
        message_calls: Sequence[FunctionCall]
        if message["role"] == "assistant":
            message_calls = message["tool_calls"] or ()
        elif message["role"] == "tool":
            message_calls = (message["tool_call"],)
        else:
            message_calls = ()

        for tool_call in message_calls:
            object_id = id(tool_call)
            if object_id in seen_call_objects:
                continue
            seen_call_objects.add(object_id)
            tool_calls.append(tool_call)
    return tuple(tool_calls)


def run_bstw_trajectory(
    user_task: GeneratedUserTask,
    injection_task: GeneratedInjectionTask,
    retry_index: int,
    config: RedTeamingConfig,
) -> Trajectory:
    """Run one independent task/injection pair in a fresh BSTW environment."""

    from agentic_red_teaming.graph import Trajectory
    from custom_suites.bstw.task_suite import task_suite

    if retry_index < 0:
        raise ValueError("retry_index cannot be negative.")

    clean_environment = task_suite.load_and_inject_default_environment({})
    attacked_environment = apply_injection_plan(clean_environment, injection_task)
    messages = run_camel_query(user_task.prompt, attacked_environment, config)
    model_output = _model_output_from_messages(messages)
    tool_calls = _tool_calls_from_messages(messages)

    fingerprint = json.dumps(
        {
            "user_task_id": user_task.task_id,
            "task_index": user_task.metadata.get("batch_index"),
            "injection_task_id": injection_task.task_id,
            "retry_index": retry_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    trajectory_digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]

    return Trajectory(
        trajectory_id=f"trajectory_{trajectory_digest}",
        retry_index=retry_index,
        messages=messages,
        model_output=model_output,
        tool_calls=tool_calls,
        task_completed=None,
        metadata={
            "user_task_id": user_task.task_id,
            "task_index": user_task.metadata.get("batch_index", 0),
            "sample_index": retry_index,
            "injection_task_id": injection_task.task_id,
            "selected_target_count": len(injection_task.selected_targets),
            "privileged_llm_model": config.privileged_llm_model,
            "quarantined_llm_model": (config.quarantined_llm_model or config.privileged_llm_model),
        },
    )


__all__ = ["apply_injection_plan", "run_bstw_trajectory", "run_camel_query"]
