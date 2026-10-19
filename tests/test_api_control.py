from __future__ import annotations

import inspect

import openai
from agentdojo.agent_pipeline.llms import openai_llm
from pydantic_ai import ModelHTTPError
from pydantic_ai.models.openai import OpenAIModel

from camel.api_control import (
    fail_fast_agentdojo_api_requests,
    is_model_api_error,
    make_fail_fast_pydantic_model,
)


def test_model_api_error_detection_follows_exception_causes() -> None:
    provider_error = ModelHTTPError(429, "gpt-test", {"error": "quota exceeded"})
    try:
        raise RuntimeError("outer") from provider_error
    except RuntimeError as wrapped_error:
        assert is_model_api_error(wrapped_error)

    assert not is_model_api_error(ValueError("ordinary plan error"))


def test_openai_pydantic_model_disables_sdk_retries(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = make_fail_fast_pydantic_model("openai:gpt-test")

    assert isinstance(model, OpenAIModel)
    assert model.client.max_retries == 0


def test_agentdojo_retry_wrapper_is_bypassed_and_restored() -> None:
    original_request = openai_llm.chat_completion_request

    with fail_fast_agentdojo_api_requests():
        assert openai_llm.chat_completion_request is inspect.unwrap(original_request)

    assert openai_llm.chat_completion_request is original_request


def test_openai_sdk_errors_are_api_errors() -> None:
    assert is_model_api_error(openai.OpenAIError("provider failure"))
