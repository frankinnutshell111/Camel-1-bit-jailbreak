"""Composite environment for Banking, Slack, Travel, and Workspace (BSTW)."""

from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.default_suites.v1.slack.task_suite import SlackEnvironment
from agentdojo.default_suites.v1.tools.calendar_client import Calendar
from agentdojo.default_suites.v1.tools.email_client import Inbox
from agentdojo.default_suites.v1.tools.types import CalendarEvent, Email
from agentdojo.default_suites.v1.travel.task_suite import TravelEnvironment
from agentdojo.default_suites.v1.workspace.task_suite import WorkspaceEnvironment


class BSTWEnvironment(
    BankingEnvironment,
    SlackEnvironment,
    WorkspaceEnvironment,
    TravelEnvironment,
):
    """Flat union of the four source environments.

    Workspace and Travel use the same Inbox and Calendar types. BSTW therefore
    merges those two pairs so their shared tools continue to resolve the normal
    ``inbox`` and ``calendar`` dependencies.
    """


def _next_numeric_id(used_ids: set[str]) -> str:
    numeric_ids = [int(item_id) for item_id in used_ids if item_id.isdigit()]
    candidate = max(numeric_ids, default=-1) + 1
    while str(candidate) in used_ids:
        candidate += 1
    return str(candidate)


def _copy_emails_with_unique_ids(
    primary: list[Email],
    secondary: list[Email],
) -> list[Email]:
    combined = [email.model_copy(deep=True) for email in primary]
    used_ids = {email.id_ for email in combined}
    for source_email in secondary:
        email = source_email.model_copy(deep=True)
        if email.id_ in used_ids:
            email.id_ = _next_numeric_id(used_ids)
        combined.append(email)
        used_ids.add(email.id_)
    return combined


def _copy_events_with_unique_ids(
    primary: list[CalendarEvent],
    secondary: list[CalendarEvent],
) -> list[CalendarEvent]:
    combined = [event.model_copy(deep=True) for event in primary]
    used_ids = {event.id_ for event in combined}
    for source_event in secondary:
        event = source_event.model_copy(deep=True)
        if event.id_ in used_ids:
            event.id_ = _next_numeric_id(used_ids)
        combined.append(event)
        used_ids.add(event.id_)
    return combined


def _merge_inboxes(workspace: Inbox, travel: Inbox) -> Inbox:
    if workspace.account_email != travel.account_email:
        raise ValueError("Workspace and Travel inboxes belong to different accounts")

    contacts_by_email = {
        contact.email.casefold(): contact.model_copy(deep=True)
        for contact in [*workspace.contact_list, *travel.contact_list]
    }
    initial_emails = _copy_emails_with_unique_ids(
        workspace.initial_emails,
        travel.initial_emails,
    )
    used_ids = {email.id_ for email in initial_emails}
    trash: dict[str, Email] = {}
    for source_inbox in (workspace, travel):
        for source_email in source_inbox.trash.values():
            email = source_email.model_copy(deep=True)
            if email.id_ in used_ids:
                email.id_ = _next_numeric_id(used_ids)
            trash[email.id_] = email
            used_ids.add(email.id_)

    return Inbox(
        account_email=workspace.account_email,
        initial_emails=initial_emails,
        contact_list=list(contacts_by_email.values()),
        trash=trash,
    )


def _merge_calendars(workspace: Calendar, travel: Calendar) -> Calendar:
    if workspace.account_email != travel.account_email:
        raise ValueError("Workspace and Travel calendars belong to different accounts")
    if workspace.current_day != travel.current_day:
        raise ValueError("Workspace and Travel calendars use different current days")

    return Calendar(
        current_day=workspace.current_day,
        account_email=workspace.account_email,
        initial_events=_copy_events_with_unique_ids(
            workspace.initial_events,
            travel.initial_events,
        ),
    )


def combine_source_environments(
    banking: BankingEnvironment,
    slack: SlackEnvironment,
    workspace: WorkspaceEnvironment,
    travel: TravelEnvironment,
) -> BSTWEnvironment:
    """Create a fresh BSTW environment containing all four source datasets."""
    return BSTWEnvironment(
        bank_account=banking.bank_account.model_copy(deep=True),
        filesystem=banking.filesystem.model_copy(deep=True),
        user_account=banking.user_account.model_copy(deep=True),
        slack=slack.slack.model_copy(deep=True),
        web=slack.web.model_copy(deep=True),
        inbox=_merge_inboxes(workspace.inbox, travel.inbox),
        calendar=_merge_calendars(workspace.calendar, travel.calendar),
        cloud_drive=workspace.cloud_drive.model_copy(deep=True),
        hotels=travel.hotels.model_copy(deep=True),
        restaurants=travel.restaurants.model_copy(deep=True),
        car_rental=travel.car_rental.model_copy(deep=True),
        flights=travel.flights.model_copy(deep=True),
        user=travel.user.model_copy(deep=True),
        reservation=travel.reservation.model_copy(deep=True),
    )
