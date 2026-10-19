"""AgentDojo task-suite definition for the index-repair experiment."""

from pathlib import Path

from agentdojo.functions_runtime import make_function
from agentdojo.task_suite import TaskSuite

from custom_suites.repair_database.environment import RedTeamingEnvironment
from custom_suites.repair_database.injection_tasks import InjectionTask0
from custom_suites.repair_database.tools import (
    get_index_diagnostics,
    get_index_status,
    rebuild_project_index,
    repair_index_shard,
)
from custom_suites.repair_database.user_tasks import UserTask0

DATA_PATH = Path(__file__).parent / "data"

TOOLS = [
    get_index_diagnostics,
    repair_index_shard,
    rebuild_project_index,
    get_index_status,
]

task_suite = TaskSuite[RedTeamingEnvironment](
    "redteaming",
    RedTeamingEnvironment,
    [make_function(tool) for tool in TOOLS],
    data_path=DATA_PATH,
    benchmark_version=(1, 2, 0),
)
task_suite.register_user_task(UserTask0)
task_suite.register_new_injection_task((1, 2, 0), InjectionTask0)
