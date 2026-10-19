"""Registration entry point for the external-email experiment suite."""

from agentdojo.task_suite import register_suite

import custom_suites.external_email.attack as _attack  # noqa: F401 - Register custom attack.
from custom_suites.external_email.task_suite import task_suite


def register_external_email_suite() -> None:
    """Register the suite for the benchmark version used by this project."""
    register_suite(task_suite, "v1.2")


register_external_email_suite()

__all__ = ["register_external_email_suite", "task_suite"]
