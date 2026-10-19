from __future__ import annotations

from dataclasses import replace

import pytest

from agentic_red_teaming.graph import (
    InjectionSelectionMode,
    RedTeamingConfig,
    TaskGenerationDifficulty,
)
from agentic_red_teaming.resources import ResourceLocation, ResourceType
from agentic_red_teaming.select_injection_targets import (
    NOT_ENOUGH_INFORMATION_INJECTION,
    select_injection_targets,
)


def _config(mode: InjectionSelectionMode) -> RedTeamingConfig:
    return RedTeamingConfig(
        difficulty=TaskGenerationDifficulty.MEDIUM,
        injection_mode=mode,
        num_retries=1,
        random_seed=42,
    )


def _locations() -> tuple[ResourceLocation, ...]:
    return (
        ResourceLocation(resource_type=ResourceType.EMAIL, resource_id="email-1"),
        ResourceLocation(resource_type=ResourceType.CALENDAR_EVENT, resource_id="event-1"),
        ResourceLocation(resource_type=ResourceType.CLOUD_FILE, resource_id="file-1"),
        ResourceLocation(resource_type=ResourceType.WEBPAGE, resource_id="page-1"),
    )


def test_none_selects_no_targets_or_payload() -> None:
    plan = select_injection_targets(_locations(), _config(InjectionSelectionMode.NONE))

    assert len(plan.candidates) == 4
    assert plan.selected_targets == ()
    assert plan.payload is None


def test_single_selection_is_seeded_and_has_fixed_payload() -> None:
    config = _config(InjectionSelectionMode.SINGLE)

    first = select_injection_targets(_locations(), config)
    second = select_injection_targets(_locations(), config)

    assert len(first.selected_targets) == 1
    assert first.selected_targets == second.selected_targets
    assert first.payload == NOT_ENOUGH_INFORMATION_INJECTION


def test_random_selection_is_seeded_and_nonempty() -> None:
    config = _config(InjectionSelectionMode.RANDOM)

    first = select_injection_targets(_locations(), config)
    second = select_injection_targets(_locations(), config)

    assert 1 <= len(first.selected_targets) <= len(first.candidates)
    assert first.selected_targets == second.selected_targets


def test_all_selects_every_unique_candidate_in_ground_truth_order() -> None:
    locations = (*_locations(), _locations()[0])

    plan = select_injection_targets(locations, _config(InjectionSelectionMode.ALL))

    assert [target.resource_id for target in plan.selected_targets] == [
        "email-1",
        "event-1",
        "file-1",
        "page-1",
    ]
    assert plan.selected_targets[0].ground_truth_resource_index == 0


def test_attack_mode_requires_a_ground_truth_resource() -> None:
    config = replace(_config(InjectionSelectionMode.NONE), injection_mode=InjectionSelectionMode.SINGLE)

    with pytest.raises(ValueError, match="requires at least one"):
        select_injection_targets((), config)
