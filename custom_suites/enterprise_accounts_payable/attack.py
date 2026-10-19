"""Two-surface attack for the Enterprise Accounts Payable Agent."""

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask

from custom_suites.enterprise_accounts_payable.injection_tasks import (
    EMAIL_INJECTION,
    SLACK_INJECTION,
    InjectionTask0,
)


@register_attack
class EnterpriseAccountsPayableConditionsAttack(BaseAttack):
    """Inject distinct payloads into the invoice email and Slack history."""

    name = "enterprise_accounts_payable_conditions"

    def attack(
        self,
        user_task: BaseUserTask,
        injection_task: BaseInjectionTask,
    ) -> dict[str, str]:
        del user_task
        if not isinstance(injection_task, InjectionTask0):
            raise ValueError(
                "Unsupported enterprise accounts-payable injection task: "
                f"{injection_task.ID}"
            )
        return {
            "unread_email_injection": EMAIL_INJECTION,
            "slack_channel_injection": SLACK_INJECTION,
        }
