"""Shared resource-location types for generated tasks and runtime injections."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field


class ResourceType(str, enum.Enum):
    """Kinds of untrusted BSTW resources whose content can be injected."""

    EMAIL = "email"
    CALENDAR_EVENT = "calendar_event"
    CLOUD_FILE = "cloud_file"
    SLACK_MESSAGE = "slack_message"
    WEBPAGE = "webpage"
    LOCAL_FILE = "local_file"


class ResourceLocation(BaseModel):
    """Stable location of one untrusted resource required by a generated task."""

    model_config = ConfigDict(frozen=True)

    resource_type: ResourceType = Field(
        description="The kind of resource containing information needed by the task."
    )
    resource_id: str = Field(
        min_length=1,
        description="The exact resource identifier copied from the resource manifest.",
    )


__all__ = ["ResourceLocation", "ResourceType"]
