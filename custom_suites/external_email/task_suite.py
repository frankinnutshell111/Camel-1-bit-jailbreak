"""AgentDojo task-suite definition for the external-email experiment."""

from pathlib import Path

from agentdojo.functions_runtime import make_function
from agentdojo.task_suite import TaskSuite

from custom_suites.external_email.environment import (
    ExternalEmailEnvironment,
)
from custom_suites.external_email.injection_tasks import (
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
)
from custom_suites.external_email.tools import (
    get_confidential_report,
    read_contact_note,
    search_contact_profiles,
    send_email,
)
from custom_suites.external_email.user_tasks import UserTask0

DATA_PATH = Path(__file__).parent / "data"

TOOLS = [
    read_contact_note,
    search_contact_profiles,
    get_confidential_report,
    send_email,
]

task_suite = TaskSuite[ExternalEmailEnvironment](
    "external_email",
    ExternalEmailEnvironment,
    [make_function(tool) for tool in TOOLS],
    data_path=DATA_PATH,
    benchmark_version=(1, 2, 0),
)
task_suite.register_user_task(UserTask0)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask0)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask1)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask2)
