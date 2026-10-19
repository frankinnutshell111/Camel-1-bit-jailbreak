"""Registration entry point for the enterprise accounts-payable suite."""

from agentdojo.task_suite import register_suite

import custom_suites.enterprise_accounts_payable.attack as _attack  # noqa: F401
from custom_suites.enterprise_accounts_payable.task_suite import task_suite


def register_enterprise_accounts_payable_suite() -> None:
    """Register the suite for the benchmark version used by this project."""
    register_suite(task_suite, "v1.2")


register_enterprise_accounts_payable_suite()

__all__ = ["register_enterprise_accounts_payable_suite", "task_suite"]
