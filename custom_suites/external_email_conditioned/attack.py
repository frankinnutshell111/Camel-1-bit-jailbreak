"""Attack mapper for the conditioned suite's independent content surfaces."""

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask

from custom_suites.external_email_conditioned.injection_tasks import (
    EXTERNAL_PROFILE_INJECTION,
    INTERNAL_NOTE_INJECTION,
    InjectionTask0,
    InjectionTask1,
    InjectionTask2,
)


@register_attack
class ExternalEmailConditionedConditionsAttack(BaseAttack):
    """Place condition-specific payloads in the intended document surfaces."""

    name = "external_email_conditioned_conditions"

    def attack(
        self,
        user_task: BaseUserTask,
        injection_task: BaseInjectionTask,
    ) -> dict[str, str]:
        del user_task
        if isinstance(injection_task, InjectionTask0):
            return {"internal_contact_note_injection": INTERNAL_NOTE_INJECTION}
        if isinstance(injection_task, InjectionTask1):
            return {"external_contact_profile_injection": EXTERNAL_PROFILE_INJECTION}
        if isinstance(injection_task, InjectionTask2):
            return {
                "internal_contact_note_injection": INTERNAL_NOTE_INJECTION,
                "external_contact_profile_injection": EXTERNAL_PROFILE_INJECTION,
            }
        raise ValueError(
            f"Unsupported conditioned external-email injection task: {injection_task.ID}"
        )
