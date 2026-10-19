# AgentDojo requires its public task-suite package to initialize before the
# versioned Workspace modules are imported.
# ruff: noqa: I001

from agentdojo.task_suite import TaskSuite, get_suite
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.default_suites.v1.tools.email_client import send_email
from agentdojo.default_suites.v1.workspace.task_suite import WorkspaceEnvironment
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str

from custom_tasks.workspace import InjectionTask14, UserTask40, register_custom_tasks


def test_register_custom_tasks_is_idempotent() -> None:
    suite = TaskSuite[WorkspaceEnvironment](
        "workspace-test",
        WorkspaceEnvironment,
        [],
        benchmark_version=(1, 2, 0),
    )

    register_custom_tasks(suite)
    register_custom_tasks(suite)

    assert list(suite.user_tasks) == ["user_task_40"]
    assert isinstance(suite.user_tasks["user_task_40"], UserTask40)
    assert list(suite.injection_tasks) == ["injection_task_14"]
    assert isinstance(suite.injection_tasks["injection_task_14"], InjectionTask14)


def test_user_task_40_ground_truth_and_utility() -> None:
    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})
    task = UserTask40()

    [call] = task.ground_truth(environment)

    assert call.function == "search_calendar_events"
    assert call.args == {
        "query": task._EVENT,
        "date": task._DATE,
    }
    assert task.utility(task.GROUND_TRUTH_OUTPUT, environment, environment)
    assert not task.utility("The event is tomorrow.", environment, environment)


def test_user_task_40_reads_networking_event_injection_vector() -> None:
    suite = get_suite("v1.2", "workspace")
    task = UserTask40()
    vector = "calendar_networking_event_injection_email"
    canary = f"<CANARY---{vector}--->"
    environment = suite.load_and_inject_default_environment({vector: canary})

    _, _, _, responses, _ = GroundTruthPipeline(task).query(
        task.PROMPT,
        FunctionsRuntime(suite.tools),
        environment,
    )

    assert any(
        response["content"] is not None
        and canary in get_text_content_as_str(response["content"])
        for response in responses
    )


def test_injection_task_14_detects_only_a_new_matching_email() -> None:
    suite = get_suite("v1.2", "workspace")
    pre_environment = suite.load_and_inject_default_environment({})
    post_environment = pre_environment.model_copy(deep=True)
    task = InjectionTask14()
    source_email = pre_environment.inbox.emails[task._SOURCE_EMAIL_ID]

    assert not task.security("", pre_environment, post_environment)

    send_email(
        post_environment.inbox,
        recipients=[task._RECIPIENT],
        subject=task._SUBJECT,
        body=f"{source_email.body}\nAltered",
    )

    assert not task.security("", pre_environment, post_environment)

    send_email(
        post_environment.inbox,
        recipients=[task._RECIPIENT],
        subject=task._SUBJECT,
        body=source_email.body,
    )

    assert task.security("", pre_environment, post_environment)


def test_injection_task_14_ground_truth_matches_goal() -> None:
    suite = get_suite("v1.2", "workspace")
    environment = suite.load_and_inject_default_environment({})
    task = InjectionTask14()

    search_call, send_call = task.ground_truth(environment)

    assert search_call.function == "search_emails"
    assert search_call.args == {"query": task._SOURCE_EMAIL_SUBJECT}
    assert send_call.function == "send_email"
    assert send_call.args == {
        "recipients": [task._RECIPIENT],
        "subject": task._SUBJECT,
        "body": environment.inbox.emails[task._SOURCE_EMAIL_ID].body,
    }
    assert send_call.placeholder_args == {
        "recipients": [task._RECIPIENT],
        "subject": task._SUBJECT,
        "body": "$email.body",
    }
