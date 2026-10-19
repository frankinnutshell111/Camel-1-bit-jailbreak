"""State models for the simulated conditioned external-email experiment."""

from agentdojo.functions_runtime import TaskEnvironment
from pydantic import BaseModel, Field


class InternalContactNote(BaseModel):
    project_id: str
    content: str


class ExternalContactProfile(BaseModel):
    full_name: str
    organization: str
    content: str


class ConfidentialReport(BaseModel):
    project_id: str
    content: str
    marker: str
    allowed_readers: set[str]


class ExperimentTargets(BaseModel):
    legitimate_recipient: str
    attacker_designated_recipient: str


class EmployeeGroupInfo(BaseModel):
    employee_number: str
    email: str
    group: str
    group_code: str
    group_email: str
    group_lead: str


class ExecutionPolicy(BaseModel):
    """Optional environment-level policy used for isolated tool testing.

    Normal CaMeL runs leave this disabled so policy replay can distinguish an
    attempted unsafe call from an allowed delivery. The authorization data
    remains present in ``ConfidentialReport.allowed_readers`` either way.
    """

    enforce_confidential_readers: bool = False


class SourceAccessLogEntry(BaseModel):
    tool: str
    arguments: dict[str, str]
    success: bool
    error: str | None = None


class EmailAttempt(BaseModel):
    to: str
    subject: str
    body: str
    contains_confidential_marker: bool
    permitted: bool
    blocked_reason: str | None = None


class BlockedEmailCall(BaseModel):
    to: str
    subject: str
    body: str
    reason: str


class SimulatedOutgoingEmail(BaseModel):
    message_id: str
    to: str
    subject: str
    body: str


class SendEmailResult(BaseModel):
    success: bool
    blocked: bool
    to: str
    message_id: str | None = None
    message: str


class EmployeeFormSubmission(BaseModel):
    submission_id: str
    name: str
    group: str
    email: str


class SubmitEmployeeFormResult(BaseModel):
    success: bool
    submission_id: str
    message: str


class ExternalEmailConditionedEnvironment(TaskEnvironment):
    internal_contact_note: InternalContactNote
    external_contact_profile: ExternalContactProfile
    confidential_report: ConfidentialReport
    experiment_targets: ExperimentTargets
    employee_group_directory: list[EmployeeGroupInfo]
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    source_access_log: list[SourceAccessLogEntry] = Field(default_factory=list)
    email_attempts: list[EmailAttempt] = Field(default_factory=list)
    blocked_email_calls: list[BlockedEmailCall] = Field(default_factory=list)
    outgoing_messages: list[SimulatedOutgoingEmail] = Field(default_factory=list)
    form_submissions: list[EmployeeFormSubmission] = Field(default_factory=list)
