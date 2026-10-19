"""Registration entry point for the conditioned external-email suite."""

from agentdojo.task_suite import register_suite

import custom_suites.external_email_conditioned.attack as _attack  # noqa: F401 - Register custom attack.
from custom_suites.external_email_conditioned.task_suite import task_suite


def register_external_email_conditioned_suite() -> None:
    """Register the suite for the benchmark version used by this project."""
    register_suite(task_suite, "v1.2")


register_external_email_conditioned_suite()

__all__ = ["register_external_email_conditioned_suite", "task_suite"]
