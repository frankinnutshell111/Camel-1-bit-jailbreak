"""Aggregate AgentDojo Banking, Slack, Travel, and Workspace into BSTW."""

from agentdojo.functions_runtime import Function
from agentdojo.task_suite import TaskSuite, get_suite

from custom_suites.bstw.data import (
    VENDORX_EMAIL_INJECTION_VECTOR,
    add_vendorx_attack_webpage,
    add_vendorx_invoice,
)
from custom_suites.bstw.environment import (
    BSTWEnvironment,
    combine_source_environments,
)
from custom_suites.bstw.injection_tasks import InjectionTask0, InjectionTask1
from custom_suites.bstw.user_tasks import UserTask0, UserTask1

SOURCE_SUITE_NAMES = ("banking", "slack", "workspace", "travel")
SOURCE_SUITES = {
    name: get_suite("v1.2", name)
    for name in SOURCE_SUITE_NAMES
}


def _collect_tools() -> list[Function]:
    tools_by_name: dict[str, Function] = {}
    for suite in SOURCE_SUITES.values():
        for tool in suite.tools:
            existing = tools_by_name.get(tool.name)
            if existing is not None and existing.run is not tool.run:
                raise ValueError(
                    f"Source suites define incompatible tools named '{tool.name}'"
                )
            tools_by_name.setdefault(tool.name, tool)
    return list(tools_by_name.values())


class BSTWTaskSuite(TaskSuite[BSTWEnvironment]):
    """TaskSuite that composes the four installed AgentDojo data fixtures."""

    def get_injection_vector_defaults(self) -> dict[str, str]:
        defaults: dict[str, str] = {}
        for suite_name, suite in SOURCE_SUITES.items():
            for vector, value in suite.get_injection_vector_defaults().items():
                if vector in defaults:
                    raise ValueError(
                        f"Duplicate injection vector '{vector}' in {suite_name}"
                    )
                defaults[vector] = value
        defaults[VENDORX_EMAIL_INJECTION_VECTOR] = ""
        return defaults

    def load_and_inject_default_environment(
        self,
        injections: dict[str, str],
    ) -> BSTWEnvironment:
        defaults = self.get_injection_vector_defaults()
        unknown = set(injections) - set(defaults)
        if unknown:
            raise ValueError(
                f"Unknown BSTW injection vectors: {sorted(unknown)}"
            )

        source_environments = {}
        for suite_name, suite in SOURCE_SUITES.items():
            source_vectors = suite.get_injection_vector_defaults()
            source_injections = {
                key: value
                for key, value in injections.items()
                if key in source_vectors
            }
            source_environments[suite_name] = (
                suite.load_and_inject_default_environment(source_injections)
            )

        environment = combine_source_environments(
            banking=source_environments["banking"],
            slack=source_environments["slack"],
            workspace=source_environments["workspace"],
            travel=source_environments["travel"],
        )
        environment = add_vendorx_invoice(
            environment,
            injections.get(VENDORX_EMAIL_INJECTION_VECTOR, ""),
        )
        return add_vendorx_attack_webpage(environment)


task_suite = BSTWTaskSuite(
    "bstw",
    BSTWEnvironment,
    _collect_tools(),
    benchmark_version=(1, 2, 0),
)
task_suite.register_user_task(UserTask0)
task_suite.register_user_task(UserTask1)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask0)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask1)
