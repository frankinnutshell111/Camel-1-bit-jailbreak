"""Fail-fast controls and classification for model-provider API failures."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any, cast

import anthropic
import httpx
import openai
import pydantic_ai
from google.genai import errors as google_api_errors
from pydantic_ai.models import Model

API_TIMEOUT_SECONDS = 180.0
"""Maximum duration of one provider request before it is treated as failed."""


def _exception_chain(exception: BaseException) -> Iterator[BaseException]:
    """Yield an exception, its nested group members, and its causal chain."""

    pending = [exception]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        yield current

        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(item for item in nested if isinstance(item, BaseException))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def is_model_api_error(exception: BaseException) -> bool:
    """Return whether an exception represents a provider or HTTP API failure."""

    api_error_types = (
        pydantic_ai.ModelHTTPError,
        pydantic_ai.UsageLimitExceeded,
        openai.OpenAIError,
        anthropic.AnthropicError,
        google_api_errors.APIError,
        httpx.HTTPError,
        TimeoutError,
        ConnectionError,
    )
    return any(isinstance(item, api_error_types) for item in _exception_chain(exception))


@cache
def make_fail_fast_pydantic_model(model: str) -> Model | str:
    """Resolve supported provider names with SDK retries disabled.

    PydanticAI's ``retries`` setting remains available for structured-output
    correction. Provider transport failures themselves are attempted only once.
    Unknown providers are returned unchanged so PydanticAI can resolve them.
    """

    provider_name, separator, model_name = model.partition(":")
    if not separator:
        return model

    if provider_name == "openai":
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=0,
            timeout=API_TIMEOUT_SECONDS,
        )
        provider = OpenAIProvider(openai_client=client)
        return OpenAIModel(cast(Any, model_name), provider=provider)

    if provider_name == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_retries=0,
            timeout=API_TIMEOUT_SECONDS,
        )
        provider = AnthropicProvider(anthropic_client=client)
        return AnthropicModel(cast(Any, model_name), provider=provider)

    if provider_name in {"google", "google-gla"}:
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.providers.google_gla import GoogleGLAProvider

        http_client = httpx.AsyncClient(timeout=API_TIMEOUT_SECONDS)
        provider = GoogleGLAProvider(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_client=http_client,
        )
        return GeminiModel(cast(Any, model_name), provider=provider)

    return model


@contextmanager
def fail_fast_agentdojo_api_requests() -> Iterator[None]:
    """Temporarily bypass AgentDojo's three-attempt provider retry wrappers."""

    from agentdojo.agent_pipeline.llms import (
        anthropic_llm,
        google_llm,
        openai_llm,
    )

    modules = (openai_llm, anthropic_llm, google_llm)
    original_requests = {module: module.chat_completion_request for module in modules}
    try:
        for module, request in original_requests.items():
            setattr(module, "chat_completion_request", inspect.unwrap(request))
        yield
    finally:
        for module, request in original_requests.items():
            setattr(module, "chat_completion_request", request)


__all__ = [
    "API_TIMEOUT_SECONDS",
    "fail_fast_agentdojo_api_requests",
    "is_model_api_error",
    "make_fail_fast_pydantic_model",
]
