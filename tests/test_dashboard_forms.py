"""GUI↔CLI parity ratchet + auto-form pure-layer tests.

No streamlit needed — specs, arg building and the page mapping are
streamlit-free, so these run in CI even without the dashboard extra.
"""

from __future__ import annotations

import pytest

from tradingbot.dashboard import forms


class TestParityRatchet:
    def test_every_cli_command_is_mapped_to_a_gui_page(self):
        """새 CLI 명령이 GUI 매핑 없이 추가되면 여기서 빨간불 (파리티 래칫)."""
        registered = set(forms.get_cli_commands())
        mapped = {c for cmds in forms.PAGE_COMMANDS.values() for c in cmds}
        mapped |= set(forms.EXCLUDED_COMMANDS)
        assert registered == mapped

    def test_sync_commands_are_a_subset_of_mapped_commands(self):
        mapped = {c for cmds in forms.PAGE_COMMANDS.values() for c in cmds}
        assert forms.SYNC_COMMANDS <= mapped


class TestParamSpecs:
    def test_specs_build_for_every_command(self):
        """24개 명령 전부 위젯 스펙 생성 성공 — 미지원 파라미터 타입 0."""
        for name, cmd in forms.get_cli_commands().items():
            specs = forms.command_param_specs(cmd)
            assert {s.kind for s in specs} <= {"str", "int", "float", "bool"}, name
            assert all(s.opt.startswith("-") for s in specs), name

    def test_worker_commands_expose_workers_param(self):
        """워커 수 선택이 GUI에 노출되는 5개 명령."""
        cmds = forms.get_cli_commands()
        for name in (
            "scan",
            "combine-scan",
            "ml-train-all",
            "ml-tune-all",
            "ml-tune-thresholds-all",
        ):
            names = {s.name for s in forms.command_param_specs(cmds[name])}
            assert "workers" in names, name

    def test_bool_flag_has_off_opt(self):
        cmds = forms.get_cli_commands()
        ws = next(s for s in forms.command_param_specs(cmds["paper"]) if s.name == "use_websocket")
        assert ws.kind == "bool"
        assert ws.opt == "--websocket"
        assert ws.off_opt == "--no-websocket"


class TestBuildCliArgs:
    def _specs(self, command: str):
        return forms.command_param_specs(forms.get_cli_commands()[command])

    def test_emits_only_non_defaults(self):
        args = forms.build_cli_args(
            self._specs("scan"), {"top_n": 15, "workers": 8, "include_train": True}
        )
        assert args == ["--top", "15", "--workers", "8", "--include-train"]

    def test_bool_true_emits_flag_and_false_emits_nothing(self):
        specs = self._specs("paper")
        assert "--websocket" in forms.build_cli_args(specs, {"use_websocket": True})
        assert "--websocket" not in forms.build_cli_args(specs, {"use_websocket": False})

    def test_required_empty_raises(self):
        with pytest.raises(forms.MissingRequiredError):
            forms.build_cli_args(self._specs("download"), {"since": " "})

    def test_true_default_bool_emits_off_flag(self):
        """--write-meta(기본 True)를 끄면 --no-write-meta 가 방출돼야 한다."""
        args = forms.build_cli_args(self._specs("ml-tune-thresholds"), {"write_meta": False})
        assert args == ["--no-write-meta"]

    def test_none_default_numeric_empty_is_unset(self):
        """None 기본 숫자 옵션: 빈 입력은 미지정, 텍스트 숫자는 정상 방출."""
        spec = forms.ParamSpec(
            name="limit",
            opt="--limit",
            off_opt=None,
            kind="int",
            default=None,
            required=False,
            help="",
        )
        assert forms.build_cli_args([spec], {"limit": ""}) == []
        assert forms.build_cli_args([spec], {"limit": "7"}) == ["--limit", "7"]

    def test_float_formats_integral_values_plainly(self):
        args = forms.build_cli_args(self._specs("backtest"), {"balance": 2_000_000.0})
        assert args == ["--balance", "2000000"]

    def test_max_workers_is_at_least_one(self):
        assert forms.max_workers() >= 1


class TestDashboardLaunchPath:
    def test_app_path_resolves_from_dashboard_package(self):
        """cli/trade.py dashboard 명령의 경로 계산 회귀 가드 (CLI 분할 재발 방지)."""
        from pathlib import Path

        from tradingbot import dashboard as dashboard_pkg

        assert (Path(dashboard_pkg.__file__ or "").parent / "app.py").exists()
