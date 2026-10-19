from __future__ import annotations

from agentic_red_teaming.task_generator import (
    _TASK_FAMILIES,
    TASK_GENERATOR_SYSTEM_PROMPT,
)


def test_task_generator_requires_non_semantic_state_changing_goals() -> None:
    normalized_prompt = " ".join(TASK_GENERATOR_SYSTEM_PROMPT.split())

    assert "NON-SEMANTIC OPERATIONAL GOAL" in normalized_prompt
    assert "at least one externally observable state change" in normalized_prompt
    assert "Do not generate a task whose final goal is informational" in normalized_prompt
    assert "Retrieval, extraction, comparison, and reasoning may appear only as intermediate steps" in normalized_prompt
    assert "message is actually sent or posted" in normalized_prompt
    assert "Do not append a secondary request to summarize, report, or explain" in normalized_prompt
    assert "three distinct state-changing goals" in normalized_prompt


def test_task_families_do_not_assign_informational_deliverables() -> None:
    serialized_families = " ".join(
        f"{family_name} {instruction}" for families in _TASK_FAMILIES.values() for family_name, instruction in families
    ).casefold()

    for forbidden_fragment in (
        "fact_finding",
        "status_or_summary",
        "extract_and_reformat",
        "compare_and_recommend",
        "audit_remediate_and_report",
        "execute_report",
    ):
        assert forbidden_fragment not in serialized_families
