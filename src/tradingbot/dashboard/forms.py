"""Auto-generated Streamlit forms for CLI commands (GUI↔CLI parity layer).

Widget specs come from click introspection of the shared Typer app, so a
new CLI option shows up in the GUI without touching this module. The
parity ratchet lives in tests/test_dashboard_forms.py: every registered
CLI command must appear in PAGE_COMMANDS (or EXCLUDED_COMMANDS) or CI
fails.

Pure parts (specs, arg building, classification) are streamlit-free;
rendering imports streamlit lazily (same convention as the CLI's lazy
heavy imports).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer.main

from tradingbot.cli import app

# ── command classification ───────────────────────────────────────────

#: Commands that finish in seconds — run inline and show console output.
SYNC_COMMANDS = frozenset({"data-list", "symbols", "status", "balance"})

#: The GUI itself — no point launching it from inside itself.
EXCLUDED_COMMANDS = frozenset({"dashboard"})

#: GUI page → CLI commands surfaced there. The parity test asserts this
#: covers every registered command, so extending the CLI forces a GUI slot.
PAGE_COMMANDS: dict[str, list[str]] = {
    "Trading": ["paper", "live", "status", "balance"],
    "Backtest": ["backtest", "optimize", "walk-forward", "scan"],
    "Combine": ["combine", "combine-scan"],
    "Pipeline": ["pipeline"],
    "ML": [
        "ml-train",
        "ml-train-all",
        "ml-backtest",
        "ml-walk-forward",
        "ml-diagnostics",
        "ml-tune",
        "ml-tune-all",
        "ml-tune-thresholds",
        "ml-tune-thresholds-all",
    ],
    "Data": ["download", "download-external", "data-list", "symbols"],
}

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

#: Selectbox sentinel meaning "leave unset, use the CLI default".
USE_DEFAULT = "(default)"


class MissingRequiredError(ValueError):
    """A required CLI option was left empty in the form."""

    def __init__(self, opt: str) -> None:
        super().__init__(f"{opt} is required")
        self.opt = opt


@dataclass(frozen=True)
class ParamSpec:
    """One CLI option distilled from click introspection."""

    name: str  # python name, e.g. "exit_"
    opt: str  # primary long option, e.g. "--exit"
    off_opt: str | None  # secondary option for bools, e.g. "--no-websocket"
    kind: str  # "str" | "int" | "float" | "bool"
    default: Any
    required: bool
    help: str


@dataclass(frozen=True)
class SubmittedForm:
    """Result of a submitted command form."""

    args: list[str]  # CLI args ready for subprocess/job spawn
    values: dict[str, Any]  # raw widget values keyed by python param name


def get_cli_commands() -> dict[str, Any]:
    """Registered CLI commands (click command objects) via the Typer app.

    Typer vendors its own click fork (``typer._click``), so everything here
    duck-types instead of isinstance-checking against standalone click.
    """
    group = typer.main.get_command(app)
    commands = getattr(group, "commands", None)
    assert isinstance(commands, dict), "Typer app did not produce a command group"
    return dict(commands)


_KIND_BY_TYPE_NAME = {"integer": "int", "float": "float", "boolean": "bool"}


def command_param_specs(command: Any) -> list[ParamSpec]:
    """Distill a click command's options into widget-ready specs."""
    specs: list[ParamSpec] = []
    for param in command.params:
        if getattr(param, "param_type_name", "") != "option" or param.name is None:
            continue
        opts = [str(o) for o in param.opts]
        long_opts = [o for o in opts if o.startswith("--")]
        opt = long_opts[0] if long_opts else opts[0]
        secondary = [str(o) for o in getattr(param, "secondary_opts", []) or []]
        off_opt = secondary[0] if secondary else None
        type_name = str(getattr(param.type, "name", "text"))
        if getattr(param, "is_flag", False) or type_name == "boolean":
            kind = "bool"
        else:
            kind = _KIND_BY_TYPE_NAME.get(type_name, "str")
        specs.append(
            ParamSpec(
                name=str(param.name),
                opt=opt,
                off_opt=off_opt,
                kind=kind,
                default=param.default,
                required=bool(param.required),
                help=str(param.help or ""),
            )
        )
    return specs


def build_cli_args(specs: list[ParamSpec], values: dict[str, Any]) -> list[str]:
    """Turn form values into CLI args, emitting only non-default values.

    Raises :class:`MissingRequiredError` when a required option is empty.
    """
    args: list[str] = []
    for spec in specs:
        value = values.get(spec.name, spec.default)
        if spec.kind == "bool":
            enabled = bool(value)
            if enabled == bool(spec.default):
                continue
            if enabled:
                args.append(spec.opt)
            elif spec.off_opt is not None:
                args.append(spec.off_opt)
            continue
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "":
            if spec.required:
                raise MissingRequiredError(spec.opt)
            continue
        if value == spec.default:
            continue
        args.extend([spec.opt, _format_value(spec, value)])
    return args


def max_workers() -> int:
    """Upper bound for the --workers slider (machine CPU count)."""
    return os.cpu_count() or 1


def render_command_form(
    command_name: str,
    *,
    key: str | None = None,
    initial: dict[str, Any] | None = None,
    submit_label: str | None = None,
    confirm_text: str | None = None,
) -> SubmittedForm | None:
    """Render a st.form for the command; return args+values when submitted.

    ``initial`` overrides widget defaults (e.g. combine template prefill).
    ``confirm_text`` adds a type-to-confirm gate: submission is rejected
    unless the user typed exactly that string (used for live trading).
    """
    import streamlit as st

    command = get_cli_commands()[command_name]
    specs = command_param_specs(command)
    form_key = key or f"form_{command_name}"
    init = initial or {}

    with st.form(form_key):
        if command.help:
            st.caption(command.help)
        values: dict[str, Any] = {}
        cols = st.columns(2)
        for i, spec in enumerate(specs):
            with cols[i % 2]:
                values[spec.name] = _render_param_widget(
                    spec, key=f"{form_key}_{spec.name}", initial=init.get(spec.name)
                )
        typed_confirm = ""
        if confirm_text is not None:
            typed_confirm = st.text_input(
                f'Type "{confirm_text}" to confirm', key=f"{form_key}_confirm"
            )
        submitted = st.form_submit_button(submit_label or f"Run {command_name}", type="primary")

    if not submitted:
        return None
    if confirm_text is not None and typed_confirm.strip() != confirm_text:
        st.error(f'Confirmation failed — type "{confirm_text}" to proceed.')
        return None
    grid = values.get("param_grid")
    if isinstance(grid, str) and grid.strip():
        try:
            json.loads(grid)
        except json.JSONDecodeError as e:
            st.error(f"--param-grid is not valid JSON: {e}")
            return None
    try:
        return SubmittedForm(args=build_cli_args(specs, values), values=values)
    except MissingRequiredError as e:
        st.error(str(e))
        return None
    except ValueError as e:  # e.g. non-numeric text for a numeric option
        st.error(f"Invalid value: {e}")
        return None


# ── widget rendering (streamlit imported lazily) ─────────────────────


def _render_param_widget(spec: ParamSpec, *, key: str, initial: Any = None) -> Any:
    import streamlit as st

    default = initial if initial is not None else spec.default
    label = spec.opt
    help_ = spec.help or None

    if spec.name == "workers":
        return st.slider(
            label,
            min_value=0,
            max_value=max_workers(),
            value=int(default or 0),
            help=f"{spec.help} — 0 = auto, max {max_workers()} = CPU count",
            key=key,
        )
    if spec.name == "strategy_name":
        names = _strategy_names()
        index = names.index(default) if default in names else 0
        return st.selectbox(label, names, index=index, help=help_, key=key)
    if spec.name == "timeframe":
        return _selectbox_with_default(st, label, TIMEFRAMES, default, help_=help_, key=key)
    if spec.name == "symbol":
        symbols = _available_symbols()
        if symbols:
            return _selectbox_with_default(st, label, symbols, default, help_=help_, key=key)
        return st.text_input(label, value=default or "", help=help_, key=key)
    if spec.name == "param_grid":
        return st.text_area(label, value=default or "", help=help_, key=key)

    if spec.kind == "bool":
        return st.toggle(label, value=bool(default), help=help_, key=key)
    if spec.kind in ("int", "float") and default is None:
        # No sentinel exists for number_input, and None→0 would silently
        # send 0 to the CLI — use free text where empty means "unset".
        return st.text_input(label, value="", help=help_, key=key)
    if spec.kind == "int":
        return st.number_input(label, value=int(default), step=1, help=help_, key=key)
    if spec.kind == "float":
        return st.number_input(label, value=float(default), help=help_, key=key)
    return st.text_input(label, value="" if default is None else str(default), help=help_, key=key)


def _selectbox_with_default(
    st: Any,
    label: str,
    options: list[str],
    default: Any,
    *,
    help_: str | None,
    key: str,
) -> str | None:
    """Selectbox that maps a None default to a '(default)' sentinel choice."""
    opts = list(options)
    if default is None:
        choice = st.selectbox(label, [USE_DEFAULT, *opts], index=0, help=help_, key=key)
        return None if choice == USE_DEFAULT else str(choice)
    if default not in opts:
        opts.insert(0, str(default))
    choice = st.selectbox(label, opts, index=opts.index(default), help=help_, key=key)
    return str(choice)


def _strategy_names() -> list[str]:
    from tradingbot.strategy.registry import get_strategy_map

    return sorted(get_strategy_map())


def _available_symbols(data_dir: Path = Path("data")) -> list[str]:
    """Symbols with downloaded data (data/BTC_KRW → BTC/KRW)."""
    if not data_dir.is_dir():
        return []
    return sorted("/".join(d.name.rsplit("_", 1)) for d in data_dir.iterdir() if d.is_dir())


def _format_value(spec: ParamSpec, value: Any) -> str:
    if spec.kind == "int":
        return str(int(value))
    if spec.kind == "float":
        f = float(value)
        return str(int(f)) if f.is_integer() else str(f)
    return str(value)
