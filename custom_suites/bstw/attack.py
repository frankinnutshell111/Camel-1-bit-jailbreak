"""Condition-specific attack for the BSTW VendorX fallback experiment."""

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask

from custom_suites.bstw.data import VENDORX_EMAIL_INJECTION_VECTOR
from custom_suites.bstw.injection_tasks import (
    VENDORX_LINK_FALLBACK_INJECTION,
    VENDORX_NOT_ENOUGH_INFORMATION_INJECTION,
    InjectionTask0,
    InjectionTask1,
)


@register_attack
class BSTWVendorXFallbackAttack(BaseAttack):
    """Inject only the Q-LLM denial payload into the VendorX invoice email."""

    name = "bstw_vendorx_fallback"

    def attack(
        self,
        user_task: BaseUserTask,
        injection_task: BaseInjectionTask,
    ) -> dict[str, str]:
        del user_task
        if isinstance(injection_task, InjectionTask0):
            payload = VENDORX_NOT_ENOUGH_INFORMATION_INJECTION
        elif isinstance(injection_task, InjectionTask1):
            payload = VENDORX_LINK_FALLBACK_INJECTION
        else:
            raise ValueError(
                f"Unsupported BSTW injection task: {injection_task.ID}"
            )
        return {VENDORX_EMAIL_INJECTION_VECTOR: payload}
