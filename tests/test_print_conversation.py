from print_conversation import messages_in_chronological_order


def message(role, content, function=None):
    value = {"role": role, "content": [{"type": "text", "content": content}]}
    if function is not None:
        value["tool_call"] = {"function": function}
    return value


def test_reorders_failed_attempt_for_display():
    user_query = message("user", "Do the task")
    tool_result = message("tool", "calendar result", "search_calendar_events")
    plan = message("assistant", "```python\nsearch_calendar_events()\n```")
    error = message("user", "Running the code gave the following error:\nNotEnoughInformationError")
    corrected_plan = message("assistant", "```python\nprint(result)\n```")

    messages = [user_query, tool_result, plan, error, corrected_plan]

    assert messages_in_chronological_order(messages) == [
        user_query,
        plan,
        tool_result,
        error,
        corrected_plan,
    ]


def test_preserves_successful_attempt_order():
    user_query = message("user", "Do the task")
    plan = message("assistant", "```python\nsearch_calendar_events()\n```")
    tool_result = message("tool", "calendar result", "search_calendar_events")
    output = message("assistant", "Done")
    messages = [user_query, plan, tool_result, output]

    assert messages_in_chronological_order(messages) == messages
