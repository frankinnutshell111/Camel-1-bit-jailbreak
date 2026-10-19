"""State models for the simulated Enterprise Accounts Payable Agent."""

from agentdojo.functions_runtime import TaskEnvironment
from pydantic import BaseModel, Field


class UnreadEmail(BaseModel):
    sender: str
    subject: str
    body: str


class EnterpriseFile(BaseModel):
    file_name: str
    content: str


class SlackMessage(BaseModel):
    sender: str
    body: str


class SlackChannelHistory(BaseModel):
    channel: str
    messages: list[SlackMessage]


class QueuedPayment(BaseModel):
    vendor: str
    amount: float
    po_number: str
    iban: str


class ExperimentTargets(BaseModel):
    vendor: str
    amount: float
    po_number: str
    legitimate_iban: str
    attacker_iban: str


class EnterpriseAccountsPayableEnvironment(TaskEnvironment):
    unread_emails: list[UnreadEmail]
    files: list[EnterpriseFile]
    slack_channel_history: list[SlackChannelHistory]
    experiment_targets: ExperimentTargets
    queued_payments: list[QueuedPayment] = Field(default_factory=list)
