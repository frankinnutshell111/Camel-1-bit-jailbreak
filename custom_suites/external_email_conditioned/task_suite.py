"""AgentDojo task-suite definition for the conditioned external-email experiment."""

from pathlib import Path

from agentdojo.functions_runtime import make_function
from agentdojo.task_suite import TaskSuite

from custom_suites.external_email_conditioned.environment import (
    ExternalEmailConditionedEnvironment,
)
from custom_suites.external_email_conditioned.injection_tasks import (
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
)
from custom_suites.external_email_conditioned.tools import (
    get_confidential_report,
    read_contact_note,
    search_contact_profiles,
    search_employee_group_info,
    send_email,
    submit_employee_form,
)
from custom_suites.external_email_conditioned.user_tasks import UserTask0, UserTask1

DATA_PATH = Path(__file__).parent / "data"

TOOLS = [
    read_contact_note,
    search_contact_profiles,
    get_confidential_report,
    search_employee_group_info,
    send_email,
    submit_employee_form,
]

task_suite = TaskSuite[ExternalEmailConditionedEnvironment](
    "external_email_conditioned",
    ExternalEmailConditionedEnvironment,
    [make_function(tool) for tool in TOOLS],
    data_path=DATA_PATH,
    benchmark_version=(1, 2, 0),
)
task_suite.register_user_task(UserTask0)
task_suite.register_user_task(UserTask1)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask0)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask1)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask2)
