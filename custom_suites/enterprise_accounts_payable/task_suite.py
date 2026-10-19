"""AgentDojo task-suite definition for the Enterprise Accounts Payable Agent."""

from pathlib import Path

from agentdojo.functions_runtime import make_function
from agentdojo.task_suite import TaskSuite

from custom_suites.enterprise_accounts_payable.environment import (
    EnterpriseAccountsPayableEnvironment,
)
from custom_suites.enterprise_accounts_payable.injection_tasks import InjectionTask0
from custom_suites.enterprise_accounts_payable.tools import (
    get_files,
    get_slack_channel_messages,
    get_unread_emails,
    list_slack_channels,
    queue_payment,
)
from custom_suites.enterprise_accounts_payable.user_tasks import UserTask0, UserTask1

DATA_PATH = Path(__file__).parent / "data"

TOOLS = [
    get_unread_emails,
    get_files,
    list_slack_channels,
    get_slack_channel_messages,
    queue_payment,
]

task_suite = TaskSuite[EnterpriseAccountsPayableEnvironment](
    "enterprise_accounts_payable",
    EnterpriseAccountsPayableEnvironment,
    [make_function(tool) for tool in TOOLS],
    data_path=DATA_PATH,
    benchmark_version=(1, 2, 0),
)
task_suite.register_user_task(UserTask0)
task_suite.register_user_task(UserTask1)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask0)
