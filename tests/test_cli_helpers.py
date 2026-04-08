"""Tests for CLI helper functions — template lookup, strategy resolution, combined building."""

from __future__ import annotations

import pytest

from tradingbot.cli import (
    COMBINE_TEMPLATES,
    _build_combined_strategy,
    _find_combine_template,
    _resolve_strategy,
)
from tradingbot.strategy.filters.registry import parse_filter_string


class TestFindCombineTemplate:
    def test_exact_match(self):
        result = _find_combine_template("Trend+RSI")
        assert result is not None
        assert result["label"] == "Trend+RSI"

    def test_case_insensitive(self):
        r1 = _find_combine_template("trend+rsi")
        r2 = _find_combine_template("TREND+RSI")
        r3 = _find_combine_template("Trend+RSI")
        assert r1 == r2 == r3

    def test_unknown_returns_none(self):
        assert _find_combine_template("NonExistentStrategy") is None

    def test_empty_string_returns_none(self):
        assert _find_combine_template("") is None

    def test_partial_match_returns_none(self):
        assert _find_combine_template("Trend") is None


class TestBuildCombinedStrategy:
    def test_returns_combined(self):
        from tradingbot.strategy.combined import CombinedStrategy

        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "BTC/KRW", "1h",
        )
        assert isinstance(strategy, CombinedStrategy)
        assert len(strategy.entry_filters) == 1
        assert len(strategy.exit_filters) == 1

    def test_sets_strategy_attrs(self):
        strategy = _build_combined_strategy(
            "rsi_oversold:30", "rsi_overbought:70", "ETH/KRW", "4h",
        )
        assert strategy.symbols == ["ETH/KRW"]
        assert strategy.timeframe == "4h"

    def test_sets_ml_filter_symbol_timeframe(self):
        strategy = _build_combined_strategy(
            "rsi_oversold:30 + lgbm_prob:0.35", "rsi_overbought:70", "SOL/KRW", "1h",
        )
        ml_filter = [f for f in strategy.entry_filters if f.name == "lgbm_prob"][0]
        assert ml_filter.symbol == "SOL/KRW"
        assert ml_filter.timeframe == "1h"

    def test_invalid_filter_raises(self):
        with pytest.raises(ValueError):
            _build_combined_strategy(
                "nonexistent_filter:30", "rsi_overbought:70", "BTC/KRW", "1h",
            )


class TestResolveStrategy:
    def test_registered_strategy(self):
        strategy, name, cls = _resolve_strategy("sma_cross", "BTC/KRW", "1h")
        assert name == "sma_cross"
        assert cls is not None
        assert strategy.timeframe == "1h"

    def test_combine_template(self):
        strategy, name, cls = _resolve_strategy("Trend+RSI", "BTC/KRW", "1h")
        assert name == "Trend+RSI"
        assert cls is None  # combined templates have no class

    def test_combine_template_case_insensitive(self):
        _, name, _ = _resolve_strategy("trend+rsi", "BTC/KRW", "1h")
        assert name == "Trend+RSI"

    def test_unknown_raises_exit(self):
        from click.exceptions import Exit

        with pytest.raises((SystemExit, Exit)):
            _resolve_strategy("NonExistentStrategy", "BTC/KRW", "1h")

    def test_sets_symbols(self):
        strategy, _, _ = _resolve_strategy(
            "Trend+RSI", "BTC/KRW", "1h", symbols=["BTC/KRW", "ETH/KRW"],
        )
        assert strategy.symbols == ["BTC/KRW", "ETH/KRW"]

    def test_sets_timeframe(self):
        strategy, _, _ = _resolve_strategy("ML+TrendEMA", "BTC/KRW", "4h")
        assert strategy.timeframe == "4h"

    def test_ml_template_rejects_multi_symbol(self):
        from click.exceptions import Exit

        with pytest.raises((SystemExit, Exit)):
            _resolve_strategy(
                "ML+TrendEMA", "BTC/KRW", "1h",
                symbols=["BTC/KRW", "ETH/KRW"],
            )

    def test_non_ml_template_allows_multi_symbol(self):
        strategy, _, _ = _resolve_strategy(
            "Trend+RSI", "BTC/KRW", "1h",
            symbols=["BTC/KRW", "ETH/KRW"],
        )
        assert strategy.symbols == ["BTC/KRW", "ETH/KRW"]


class TestCombineTemplates:
    def test_all_templates_parse_successfully(self):
        """Every template's entry and exit strings must parse without error."""
        for tmpl in COMBINE_TEMPLATES:
            try:
                parse_filter_string(tmpl["entry"])
                parse_filter_string(tmpl["exit"])
            except Exception as e:
                pytest.fail(f"Template '{tmpl['label']}' failed to parse: {e}")

    def test_unique_labels(self):
        labels = [t["label"].lower() for t in COMBINE_TEMPLATES]
        assert len(labels) == len(set(labels)), f"Duplicate labels: {[l for l in labels if labels.count(l) > 1]}"

    def test_required_keys(self):
        for tmpl in COMBINE_TEMPLATES:
            assert "entry" in tmpl, f"Missing 'entry' in {tmpl}"
            assert "exit" in tmpl, f"Missing 'exit' in {tmpl}"
            assert "label" in tmpl, f"Missing 'label' in {tmpl}"

    def test_ml_templates_use_threshold_035(self):
        """All ML templates should use 0.35 threshold (veto mode)."""
        for tmpl in COMBINE_TEMPLATES:
            entry = tmpl["entry"]
            if "lgbm_prob" in entry:
                assert "lgbm_prob:0.35" in entry, (
                    f"Template '{tmpl['label']}' uses wrong ML threshold: {entry}"
                )
