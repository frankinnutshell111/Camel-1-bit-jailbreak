#!/usr/bin/env python3
"""Visualize one streamed agentic red-teaming trajectory JSON log."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from print_conversation import content_to_text, messages_in_chronological_order

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n?(?P<code>.*?)```",
    re.DOTALL,
)


def _load_record(path: Path) -> dict[str, Any]:
    """Load and validate the streamed trajectory wrapper."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Trajectory log does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Trajectory log is not valid JSON: {path}") from exc

    if not isinstance(value, dict):
        raise ValueError("Trajectory log must contain a JSON object.")
    trajectory = value.get("trajectory")
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
        raise ValueError("Expected the streamed format with trajectory.messages at the top level.")
    return value


def _string(value: Any, default: str = "N/A") -> str:
    """Render a scalar compactly for metadata panels."""

    if value is None:
        return default
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _summary_table(record: Mapping[str, Any]) -> Table:
    """Build the run and outcome summary shown above the conversation."""

    user_task = record.get("user_task") or {}
    injection_task = record.get("injection_task") or {}
    trajectory = record.get("trajectory") or {}
    trajectory_metadata = trajectory.get("metadata") or {}
    evaluation = record.get("evaluation")

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    rows = (
        ("Report", record.get("report_id")),
        ("Status", record.get("status")),
        ("Trajectory", trajectory.get("trajectory_id")),
        ("Task", user_task.get("task_id")),
        ("Difficulty", user_task.get("difficulty")),
        ("Sample index", trajectory_metadata.get("sample_index")),
        ("P-LLM", trajectory_metadata.get("privileged_llm_model")),
        ("Q-LLM", trajectory_metadata.get("quarantined_llm_model")),
        ("Injection mode", injection_task.get("mode")),
        (
            "Injected targets",
            trajectory_metadata.get(
                "selected_target_count",
                len(injection_task.get("selected_targets") or []),
            ),
        ),
        (
            "Jailbreak",
            None if evaluation is None else evaluation.get("jailbreak"),
        ),
    )
    for label, value in rows:
        value_text = Text(_string(value))
        if label == "Jailbreak" and value is not None:
            value_text.stylize("bold red" if value else "bold green")
        table.add_row(label, value_text)
    return table


def _injection_panel(injection_task: Mapping[str, Any]) -> Panel:
    """Render selected targets and the runtime injection payload."""

    selected_targets = injection_task.get("selected_targets") or []
    lines: list[str] = []
    if selected_targets:
        lines.append("Selected targets:")
        for target in selected_targets:
            if not isinstance(target, Mapping):
                lines.append(f"  - {target}")
                continue
            resource_type = _string(target.get("resource_type"))
            resource_id = _string(target.get("resource_id"))
            vector = _string(target.get("injection_vector"), default="")
            suffix = f" — {vector}" if vector else ""
            lines.append(f"  - {resource_type}:{resource_id}{suffix}")
    else:
        lines.append("Selected targets: none")

    payload = injection_task.get("payload")
    if payload:
        lines.extend(("", "Payload:", str(payload).strip()))
    return Panel(
        Text("\n".join(lines)),
        title=Text("Injection", style="bold yellow"),
        border_style="yellow",
        padding=(1, 2),
    )


def _assistant_renderable(content: str) -> RenderableType:
    """Render fenced code as syntax and surrounding text literally."""

    elements: list[RenderableType] = []
    cursor = 0
    for match in _CODE_BLOCK_PATTERN.finditer(content):
        before = content[cursor : match.start()]
        if before:
            elements.append(Text(before))
        language = match.group("language") or "python"
        elements.append(
            Syntax(
                match.group("code").strip(),
                language,
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        )
        cursor = match.end()
    trailing = content[cursor:]
    if trailing:
        elements.append(Text(trailing))
    if not elements:
        return Text(content)
    return Group(*elements)


def _tool_renderable(message: Mapping[str, Any], content: str) -> RenderableType:
    """Show the tool arguments before its serialized result."""

    tool_call = message.get("tool_call") or {}
    arguments = tool_call.get("args") if isinstance(tool_call, Mapping) else None
    elements: list[RenderableType] = []
    if arguments is not None:
        elements.extend(
            (
                Text("Arguments", style="bold magenta"),
                Syntax(
                    json.dumps(arguments, indent=2, ensure_ascii=False, default=str),
                    "json",
                    theme="monokai",
                    word_wrap=True,
                ),
                Text("Result", style="bold magenta"),
            )
        )
    elements.append(Text(content))
    return Group(*elements)


def _message_panel(message: Mapping[str, Any]) -> Panel:
    """Build one role-colored conversation panel."""

    role = _string(message.get("role"), default="unknown")
    content = content_to_text(message.get("content", ""))
    if role == "user":
        title = Text("User", style="bold blue")
        body: RenderableType = Text(content)
        border_style = "cyan"
    elif role == "assistant":
        title = Text("Assistant / P-LLM plan", style="bold green")
        body = _assistant_renderable(content)
        border_style = "green"
    elif role == "tool":
        tool_call = message.get("tool_call") or {}
        function = tool_call.get("function", "unknown") if isinstance(tool_call, Mapping) else "unknown"
        title = Text(f"Tool — {function}", style="bold magenta")
        body = _tool_renderable(message, content)
        border_style = "magenta"
    else:
        title = Text(role.capitalize(), style="bold yellow")
        body = Text(content)
        border_style = "yellow"
    return Panel(
        body,
        title=title,
        border_style=border_style,
        padding=(1, 2),
    )


def _evaluation_panel(evaluation: Mapping[str, Any] | None) -> Panel:
    """Render the aggregate and per-plan evaluator judgments."""

    if evaluation is None:
        return Panel(
            Text("This trajectory has not been evaluated yet."),
            title=Text("Evaluation pending", style="bold yellow"),
            border_style="yellow",
        )

    jailbreak = bool(evaluation.get("jailbreak"))
    heading = Text()
    heading.append("Jailbreak: ", style="bold")
    heading.append(
        "TRUE" if jailbreak else "FALSE",
        style="bold red" if jailbreak else "bold green",
    )
    heading.append(
        "\nDramatic deviation: "
        f"{_string(evaluation.get('deviated_from_task'))}"
        " | Unsafe risk: "
        f"{_string(evaluation.get('unsafe_risk'))}"
    )
    heading.append(f"\n\nRationale: {_string(evaluation.get('rationale'))}")

    plan_table = Table(
        "Plan",
        "Jailbreak",
        "Deviation",
        "Unsafe",
        "Explanation",
        expand=True,
        show_lines=True,
    )
    for plan in evaluation.get("plan_evaluations") or []:
        if not isinstance(plan, Mapping):
            continue
        plan_table.add_row(
            _string(plan.get("plan_index")),
            _string(plan.get("jailbreak")),
            _string(plan.get("dramatic_deviation")),
            _string(plan.get("unsafe_risk")),
            _string(plan.get("explanation")),
        )
    body: RenderableType = Group(heading, Text(""), plan_table) if plan_table.row_count else heading
    return Panel(
        body,
        title=Text("Trajectory evaluation", style="bold red" if jailbreak else "bold green"),
        border_style="red" if jailbreak else "green",
        padding=(1, 2),
    )


def visualize_trajectory_log(
    json_file_path: str | Path,
    *,
    console: Console | None = None,
    use_pager: bool = True,
) -> None:
    """Render one current-format trajectory log with Rich."""

    path = Path(json_file_path)
    record = _load_record(path)
    console = console or Console()
    pager = console.pager() if use_pager else nullcontext()

    with pager:
        console.print(
            Panel(
                _summary_table(record),
                title=Text("Agentic red-teaming trajectory", style="bold blue"),
                border_style="blue",
            )
        )
        console.print()

        user_task = record.get("user_task") or {}
        console.print(
            Panel(
                Text(_string(user_task.get("prompt"), default="")),
                title=Text("Generated user task", style="bold blue"),
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()
        console.print(_injection_panel(record.get("injection_task") or {}))
        console.print()

        console.rule("[bold blue]Conversation (chronological execution order)[/bold blue]")
        messages: Sequence[Mapping[str, Any]] = record["trajectory"]["messages"]
        for message in messages_in_chronological_order(list(messages)):
            console.print(_message_panel(message))
            console.print()

        console.rule("[bold blue]Evaluation[/bold blue]")
        console.print(_evaluation_panel(record.get("evaluation")))
        console.print()
        console.rule("[bold blue]End of trajectory[/bold blue]")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", help="Path to one trajectories/*.json file.")
    parser.add_argument(
        "--no-pager",
        action="store_true",
        help="Print directly instead of opening the terminal pager.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = _parse_args(argv)
    try:
        visualize_trajectory_log(args.json_file, use_pager=not args.no_pager)
    except ValueError as exc:
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
