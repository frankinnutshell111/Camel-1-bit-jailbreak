"""Registration entry point for the red-teaming task suite."""

from agentdojo.task_suite import register_suite

from custom_suites.repair_database.task_suite import task_suite


def register_redteaming_suite() -> None:
    """Register the suite for the benchmark version used by this project."""
    register_suite(task_suite, "v1.2")


register_redteaming_suite()

__all__ = ["register_redteaming_suite", "task_suite"]

