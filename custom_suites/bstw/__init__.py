"""Registration entry point for the aggregate BSTW suite."""

from agentdojo.task_suite import register_suite

import custom_suites.bstw.attack as _attack  # noqa: F401 - Register custom attack.
from custom_suites.bstw.task_suite import task_suite


def register_bstw_suite() -> None:
    """Register BSTW for the benchmark version used by this project."""
    register_suite(task_suite, "v1.2")


register_bstw_suite()

__all__ = ["register_bstw_suite", "task_suite"]
