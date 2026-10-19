# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast

import pytest
from pydantic_ai import ModelHTTPError

from camel.capabilities import Capabilities
from camel.interpreter import interpreter, result, value
from camel.interpreter import namespace as ns
from camel.security_policy import NoSecurityPolicyEngine


def test_exception():
    code = "raise ValueError('Error!')"
    namespace = ns.Namespace.with_builtins()
    got_code, got_namespace, _, _ = interpreter.camel_eval(
        ast.parse(code),
        namespace,
        [],
        [],
        interpreter.EvalArgs(NoSecurityPolicyEngine(), interpreter.MetadataEvalMode.NORMAL),
    )
    assert isinstance(got_code, result.Error)
    assert ValueError == type(got_code.error.exception)
    assert "Error!" == str(got_code.error.exception)
    assert got_namespace == namespace


def test_qllm_api_error_escapes_interpreter() -> None:
    def query_ai_assistant() -> str:
        raise ModelHTTPError(429, "gpt-test", {"error": "quota exceeded"})

    namespace = ns.Namespace(
        {
            "query_ai_assistant": value.CaMeLBuiltin(
                "query_ai_assistant",
                query_ai_assistant,
                Capabilities.camel(),
                (),
            )
        }
    )

    with pytest.raises(ModelHTTPError, match="status_code: 429"):
        interpreter.camel_eval(
            ast.parse("query_ai_assistant()"),
            namespace,
            [],
            [],
            interpreter.EvalArgs(
                NoSecurityPolicyEngine(),
                interpreter.MetadataEvalMode.NORMAL,
            ),
        )


def test_non_api_tool_error_remains_an_interpreter_error() -> None:
    def query_ai_assistant() -> str:
        raise ValueError("invalid output schema")

    namespace = ns.Namespace(
        {
            "query_ai_assistant": value.CaMeLBuiltin(
                "query_ai_assistant",
                query_ai_assistant,
                Capabilities.camel(),
                (),
            )
        }
    )

    evaluation, _, _, _ = interpreter.camel_eval(
        ast.parse("query_ai_assistant()"),
        namespace,
        [],
        [],
        interpreter.EvalArgs(
            NoSecurityPolicyEngine(),
            interpreter.MetadataEvalMode.NORMAL,
        ),
    )

    assert isinstance(evaluation, result.Error)
    assert isinstance(evaluation.error.exception, ValueError)
