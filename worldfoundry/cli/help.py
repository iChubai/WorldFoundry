"""Terminal-aware help rendering for the public WorldFoundry CLIs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

_RESET = "\033[0m"
_ACCENT = "\033[38;5;100m"
_HEADING = "\033[1m"
_MUTED = "\033[2m"
_DEFAULT = "\033[38;5;100m"


def _color_enabled() -> bool:
    """Return whether interactive ANSI help should be emitted."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None:
        return force.lower() not in {"", "0", "false", "no"}
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _paint(text: str, style: str) -> str:
    if not text or not style or not _color_enabled():
        return text
    return f"{style}{text}{_RESET}"


def _terminal_help_enabled() -> bool:
    """Keep the panel layout in a colorless TTY, and plain help in pipes."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "╭─╮│╰╯".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return _color_enabled() or (
        os.environ.get("TERM") != "dumb" and bool(getattr(sys.stdout, "isatty", lambda: False)())
    )


@dataclass(frozen=True)
class _HelpRow:
    invocation: str
    description: str
    default: str | None = None


@dataclass(frozen=True)
class _HelpPanel:
    title: str
    rows: tuple[_HelpRow, ...]
    description: str = ""


class WorldFoundryArgumentParser(argparse.ArgumentParser):
    """Argument parser with compact rich help in interactive terminals.

    Parsing and non-interactive help remain standard ``argparse`` behavior.
    This keeps redirects, docs generation, shell completion, and tests stable.
    """

    def format_help(self) -> str:
        if not _terminal_help_enabled():
            return super().format_help()
        return self._format_terminal_help()

    def _format_terminal_help(self) -> str:
        terminal_width = max(20, min(shutil.get_terminal_size((120, 24)).columns, 200))
        usage = _wrap_usage(self._terminal_usage(), terminal_width)
        if usage.startswith("usage:"):
            usage = _paint("usage:", _HEADING) + usage[len("usage:") :]

        blocks = [usage]
        if self.description:
            blocks.append(_wrap_usage(str(self.description).strip(), terminal_width))

        panels = self._help_panels()
        rendered = _render_panel_grid(panels, terminal_width)
        if rendered:
            blocks.append(rendered)

        if self.epilog:
            blocks.append(_render_epilog(str(self.epilog), terminal_width))
        return "\n\n".join(block for block in blocks if block).rstrip() + "\n"

    def _terminal_usage(self) -> str:
        if self.usage is not None:
            return super().format_usage().strip()
        formatter = self._get_formatter()
        actions = [action for action in self._actions if action.help != argparse.SUPPRESS]
        required_groups = [group for group in self._mutually_exclusive_groups if group.required]
        required = [
            action
            for action in actions
            if action.option_strings
            and (action.required or any(action in group._group_actions for group in required_groups))
        ]
        parts = [f"usage: {self.prog}"]
        if required:
            parts.append(formatter._format_actions_usage(required, required_groups))
        if any(action.option_strings and action not in required for action in actions):
            parts.append("[OPTIONS]")
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                parts.append("COMMAND ..." if action.required else "[COMMAND ...]")
            elif not action.option_strings:
                parts.append(formatter._format_actions_usage([action], []))
        return " ".join(parts)

    def _help_panels(self) -> tuple[_HelpPanel, ...]:
        panels: list[_HelpPanel] = []
        additional_panels: list[_HelpPanel] = []
        for group in self._action_groups:
            actions = tuple(action for action in group._group_actions if action.help != argparse.SUPPRESS)
            if not actions:
                continue
            if group.title == "options" and len(actions) > 12:
                # Keep explicitly named pipeline groups near the top of model help.
                for panel in self._split_options(actions):
                    (panels if panel.title == "options" else additional_panels).append(panel)
                continue
            ordinary_actions = [action for action in actions if not isinstance(action, argparse._SubParsersAction)]
            if ordinary_actions:
                panels.extend(
                    self._split_options(
                        ordinary_actions, title=str(group.title), description=str(group.description or "")
                    )
                )
            for action in actions:
                if not isinstance(action, argparse._SubParsersAction):
                    continue
                commands = tuple(action._get_subactions())
                rows = tuple(self._help_row(command) for command in commands if command.help != argparse.SUPPRESS)
                # argparse only creates help pseudo-actions for commands with help=.
                if not commands:
                    rows = tuple(
                        _HelpRow(name, str(parser.description or "")) for name, parser in action.choices.items()
                    )
                if rows:
                    panels.append(_HelpPanel("commands", rows, str(group.description or "")))
        return tuple(panels + additional_panels)

    def _split_options(
        self, actions: Sequence[argparse.Action], *, title: str | None = None, description: str = ""
    ) -> tuple[_HelpPanel, ...]:
        buckets: dict[str, list[argparse.Action]] = {}
        for action in actions:
            dotted_prefix = next(
                (
                    option[2:].rsplit(".", 1)[0]
                    for option in action.option_strings
                    if option.startswith("--") and "." in option
                ),
                None,
            )
            section = f"{dotted_prefix} options" if dotted_prefix else title or _option_section(action)
            buckets.setdefault(section, []).append(action)
        return tuple(
            _HelpPanel(
                section, tuple(self._help_row(action) for action in section_actions), description if index == 0 else ""
            )
            for index, (section, section_actions) in enumerate(buckets.items())
        )

    def _help_row(self, action: argparse.Action) -> _HelpRow:
        formatter = self._get_formatter()
        invocation = formatter._format_action_invocation(action)
        description = formatter._expand_help(action) if action.help else ""
        default = _format_default(action.default) if _should_show_default(action, description) else None
        return _HelpRow(invocation=invocation, description=description, default=default)


def _wrap_usage(usage: str, width: int) -> str:
    """Keep argparse choices and long dotted options inside the terminal width."""
    lines: list[str] = []
    for line in usage.splitlines():
        if len(line) <= width:
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=indent,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def _option_section(action: argparse.Action) -> str:
    dest = action.dest.replace("-", "_")
    if dest in {"help", "json", "output", "verbose", "quiet"} or dest.startswith("output_"):
        return "options"
    if dest.startswith("generation_cache") or "cache" in dest:
        return "cache options"
    if dest.startswith(("model", "ckpt")) or dest in {"device", "gpu", "dtype", "low_vram"}:
        return "model options"
    if dest.startswith(("benchmark", "suite", "task", "metric", "dataset")):
        return "benchmark and task options"
    if dest.startswith(("plan", "resume", "engine", "mode", "timeout", "workdir", "env", "fail_", "skip_")):
        return "execution options"
    if dest.startswith(("input", "request", "result", "artifact", "data_", "prompt", "seed", "frame", "step")):
        return "input and generation options"
    return "additional options"


def _should_show_default(action: argparse.Action, description: str) -> bool:
    if "%(default)" in str(action.help) or "(default:" in description:
        return False
    if action.default is None or action.default == argparse.SUPPRESS:
        return False
    if action.dest == "help" or action.required:
        return False
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return False
    return True


def _format_default(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _render_panel_grid(panels: Sequence[_HelpPanel], terminal_width: int) -> str:
    if not panels:
        return ""
    if terminal_width < 112 or len(panels) == 1:
        return "\n\n".join(_render_panel(panel, terminal_width) for panel in panels)

    gap = 2
    left_width = (terminal_width - gap) // 2
    right_width = terminal_width - gap - left_width
    columns: list[list[str]] = [[], []]
    heights = [0, 0]
    # Command catalogs can dwarf all other panels. Split at command boundaries
    # so both columns stay useful without hiding any entries.
    balanced_panels = []
    for panel in panels:
        if panel.title == "commands" and len(panel.rows) > 10:
            for start in range(0, len(panel.rows), 10):
                balanced_panels.append(
                    _HelpPanel(
                        "commands" if start == 0 else "commands (continued)",
                        panel.rows[start : start + 10],
                        panel.description if start == 0 else "",
                    )
                )
        else:
            balanced_panels.append(panel)
    for panel in balanced_panels:
        column_index = 0 if heights[0] <= heights[1] else 1
        panel_width = left_width if column_index == 0 else right_width
        panel_lines = _render_panel_lines(panel, panel_width)
        if columns[column_index]:
            columns[column_index].append(" " * panel_width)
            heights[column_index] += 1
        columns[column_index].extend(panel_lines)
        heights[column_index] += len(panel_lines)

    output: list[str] = []
    for row_index in range(max(heights)):
        left_line = columns[0][row_index] if row_index < heights[0] else " " * left_width
        right_line = columns[1][row_index] if row_index < heights[1] else ""
        output.append(left_line + " " * gap + right_line)
    return "\n".join(output)


def _render_panel(panel: _HelpPanel, width: int) -> str:
    return "\n".join(_render_panel_lines(panel, width))


def _render_panel_lines(panel: _HelpPanel, width: int) -> list[str]:
    inner_width = width - 4
    title = f" {panel.title} "
    # Long dotted group names move inside the box on small terminals.
    title_fits = len(title) <= width - 4
    top = f"╭─{title}{'─' * (width - len(title) - 3)}╮" if title_fits else f"╭{'─' * (width - 2)}╮"
    lines = [_paint(top, _ACCENT)]

    def append_line(text: str = "", style: str = "") -> None:
        lines.append(
            _paint("│", _ACCENT)
            + " "
            + _paint(text, style)
            + " " * (inner_width - len(text))
            + " "
            + _paint("│", _ACCENT)
        )

    if not title_fits:
        for title_line in textwrap.wrap(panel.title, width=inner_width, break_on_hyphens=False):
            append_line(title_line, _ACCENT)

    for description_line in textwrap.wrap(
        panel.description,
        width=inner_width,
        break_long_words=True,
        break_on_hyphens=False,
    ):
        append_line(description_line)
    if panel.description or not title_fits:
        append_line("─" * inner_width, _ACCENT)

    for index, row in enumerate(panel.rows):
        if index and panel.title.startswith("commands"):
            append_line()
        invocation_lines = textwrap.wrap(
            row.invocation,
            width=inner_width,
            subsequent_indent="  ",
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for invocation in invocation_lines:
            append_line(invocation, _HEADING)
        for description, visible_length in _description_lines(row, inner_width - 2):
            lines.append(
                _paint("│", _ACCENT)
                + "   "
                + description
                + " " * (inner_width - 2 - visible_length)
                + " "
                + _paint("│", _ACCENT)
            )
    lines.append(_paint(f"╰{'─' * (width - 2)}╯", _ACCENT))
    return lines


def _description_lines(row: _HelpRow, width: int) -> list[tuple[str, int]]:
    description = " ".join(row.description.split())
    if row.default is not None:
        description = f"{description} (default: {row.default})".strip()
    default_start = description.find("(default:")
    offset = 0
    lines = []
    for line in textwrap.wrap(description, width=width, break_on_hyphens=False):
        offset = description.index(line, offset)
        split = len(line) if default_start < 0 else max(0, min(len(line), default_start - offset))
        lines.append((_paint(line[:split], _MUTED) + _paint(line[split:], _DEFAULT), len(line)))
        offset += len(line)
    return lines


def _render_epilog(epilog: str, width: int) -> str:
    rendered: list[str] = []
    for line in epilog.strip().splitlines():
        if not line:
            rendered.append("")
            continue
        if line.endswith(":"):
            rendered.append(_paint(line, _HEADING))
        else:
            rendered.extend(_paint(item, _MUTED) for item in textwrap.wrap(line, width=width))
    return "\n".join(rendered)
