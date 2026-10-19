from __future__ import annotations

from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatUserMessage,
    text_content_block_from_string,
)

from agentic_red_teaming.graph import (
    GeneratedUserTask,
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
)
from agentic_red_teaming.resources import ResourceLocation, ResourceType
from agentic_red_teaming.run_bstw_trajectory import (
    apply_injection_plan,
    run_bstw_trajectory,
)
from agentic_red_teaming.select_injection_targets import (
    NOT_ENOUGH_INFORMATION_INJECTION,
    select_injection_targets,
)
from agentic_red_teaming.task_generator import build_resource_manifest
from custom_suites.bstw.environment import BSTWEnvironment
from custom_suites.bstw.task_suite import task_suite


def _config(mode: InjectionSelectionMode) -> RedTeamingConfig:
    return RedTeamingConfig(
        difficulty=TaskGenerationDifficulty.MEDIUM,
        injection_mode=mode,
        num_retries=1,
        random_seed=42,
    )


def _first_location_by_type() -> tuple[ResourceLocation, ...]:
    environment = task_suite.load_and_inject_default_environment({})
    manifest = build_resource_manifest(environment)
    locations: dict[ResourceType, ResourceLocation] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            resource_type = value.get("resource_type")
            resource_id = value.get("resource_id")
            if isinstance(resource_type, str) and isinstance(resource_id, str):
                parsed_type = ResourceType(resource_type)
                locations.setdefault(
                    parsed_type,
                    ResourceLocation(
                        resource_type=parsed_type,
                        resource_id=resource_id,
                    ),
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    return tuple(locations[resource_type] for resource_type in ResourceType)


def _resource_content(
    environment: BSTWEnvironment,
    location: ResourceLocation,
) -> str:
    if location.resource_type is ResourceType.EMAIL:
        return environment.inbox.emails[location.resource_id].body
    if location.resource_type is ResourceType.CALENDAR_EVENT:
        return environment.calendar.events[location.resource_id].description
    if location.resource_type is ResourceType.CLOUD_FILE:
        return environment.cloud_drive.files[location.resource_id].content
    if location.resource_type is ResourceType.SLACK_MESSAGE:
        mailbox_name, destination, raw_index = location.resource_id.split(":", maxsplit=2)
        return getattr(environment.slack, mailbox_name)[destination][int(raw_index)].body
    if location.resource_type is ResourceType.WEBPAGE:
        return environment.web.web_content[location.resource_id]
    if location.resource_type is ResourceType.LOCAL_FILE:
        return environment.filesystem.files[location.resource_id]
    raise AssertionError(f"Unhandled resource type: {location.resource_type}")


def test_apply_injection_plan_copies_and_injects_every_resource_type() -> None:
    clean_environment = task_suite.load_and_inject_default_environment({})
    locations = _first_location_by_type()
    injection_task = select_injection_targets(
        locations,
        _config(InjectionSelectionMode.ALL),
    )

    attacked_environment = apply_injection_plan(clean_environment, injection_task)

    for location in locations:
        assert NOT_ENOUGH_INFORMATION_INJECTION not in _resource_content(
            clean_environment,
            location,
        )
        assert NOT_ENOUGH_INFORMATION_INJECTION.strip() in _resource_content(
            attacked_environment,
            location,
        )


def test_run_bstw_trajectory_returns_the_camel_transcript(monkeypatch) -> None:
    messages: tuple[ChatMessage, ...] = (
        ChatUserMessage(
            role="user",
            content=[text_content_block_from_string("Test query")],
        ),
        ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string("Finished")],
            tool_calls=None,
        ),
    )

    def fake_run_camel_query(*_args, **_kwargs) -> tuple[ChatMessage, ...]:
        return messages

    monkeypatch.setattr(
        "agentic_red_teaming.run_bstw_trajectory.run_camel_query",
        fake_run_camel_query,
    )
    config = _config(InjectionSelectionMode.NONE)
    user_task = GeneratedUserTask(
        task_id="generated_user_task_test",
        prompt="Test query",
        difficulty=TaskGenerationDifficulty.MEDIUM,
        ground_truth=(
            ResourceLocation(resource_type=ResourceType.EMAIL, resource_id="0"),
        ),
    )
    injection_task = select_injection_targets(user_task.ground_truth, config)

    trajectory = run_bstw_trajectory(user_task, injection_task, 0, config)

    assert trajectory.messages == messages
    assert trajectory.model_output == "Finished"
    assert trajectory.task_completed is None
