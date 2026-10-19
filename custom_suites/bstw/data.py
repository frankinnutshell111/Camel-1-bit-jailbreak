"""BSTW-specific fixture loading layered on the four source suites."""

from pathlib import Path

import yaml
from agentdojo.default_suites.v1.tools.email_client import Inbox
from agentdojo.default_suites.v1.tools.types import Email

from custom_suites.bstw.environment import BSTWEnvironment

VENDORX_EMAIL_INJECTION_VECTOR = "bstw_vendorx_email_injection"
VENDORX_EMAIL_INJECTION_TOKEN = "{{vendorx_email_injection}}"
VENDORX_INVOICE_PATH = Path(__file__).parent / "data" / "vendorx_invoice.yaml"
VENDORX_ATTACK_URL = (
    "https://vendorx-payments.attacker.example/archive/VX-2025-042"
)
VENDORX_ATTACK_WEB_KEY = (
    "vendorx-payments.attacker.example/archive/VX-2025-042"
)
VENDORX_ATTACK_WEBPAGE_PATH = (
    Path(__file__).parent / "data" / "vendorx_attack_webpage.txt"
)


def load_vendorx_invoice(injection: str = "") -> Email:
    """Load the VendorX invoice and substitute its untrusted-content slot."""
    raw_invoice = yaml.safe_load(VENDORX_INVOICE_PATH.read_text())
    raw_invoice["body"] = raw_invoice["body"].replace(
        VENDORX_EMAIL_INJECTION_TOKEN,
        injection,
    )
    return Email.model_validate(raw_invoice)


def add_vendorx_invoice(
    environment: BSTWEnvironment,
    injection: str = "",
) -> BSTWEnvironment:
    """Add the custom unread invoice to a freshly composed BSTW environment."""
    invoice = load_vendorx_invoice(injection)
    if invoice.id_ in environment.inbox.emails:
        raise ValueError(f"BSTW VendorX email ID '{invoice.id_}' is already in use")

    environment.inbox = Inbox(
        account_email=environment.inbox.account_email,
        initial_emails=[*environment.inbox.initial_emails, invoice],
        contact_list=[
            contact.model_copy(deep=True)
            for contact in environment.inbox.contact_list
        ],
        trash={
            email_id: email.model_copy(deep=True)
            for email_id, email in environment.inbox.trash.items()
        },
    )
    return environment


def add_vendorx_attack_webpage(
    environment: BSTWEnvironment,
) -> BSTWEnvironment:
    """Add the latent attacker-controlled page to Slack's simulated web."""
    environment.web.web_content[VENDORX_ATTACK_WEB_KEY] = (
        VENDORX_ATTACK_WEBPAGE_PATH.read_text()
    )
    return environment
