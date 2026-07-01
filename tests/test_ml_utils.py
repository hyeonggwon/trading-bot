"""Unit tests for ML sizing helpers (Half-Kelly).

Locks the Half-Kelly contract that feeds Signal.strength: bounded to [0, 0.5],
zero at/below breakeven, and monotonically increasing in win probability. The
engines multiply position size by this value, so a regression here silently
mis-sizes real-money orders.
"""

from __future__ import annotations

import pytest

from tradingbot.ml.utils import HALF_KELLY_MAX, half_kelly, kelly_strength


def test_half_kelly_known_values():
    # full_kelly = (p*b - q)/b ; half = 0.5 * full_kelly
    # p=0.6, b=1.5: (0.9 - 0.4)/1.5 = 0.3333 → half = 0.16667
    assert half_kelly(0.6, 1.5) == pytest.approx(0.5 * (0.5 / 1.5))
    # p=0.5, b=1.5: (0.75 - 0.5)/1.5 = 0.16667 → half = 0.08333
    assert half_kelly(0.5, 1.5) == pytest.approx(0.5 * (0.25 / 1.5))


def test_half_kelly_zero_at_or_below_breakeven():
    # p=0.4, b=1.5: p*b = 0.6 == q=0.6 → edge ~0 (float dust) → no meaningful bet
    assert half_kelly(0.4, 1.5) == pytest.approx(0.0, abs=1e-12)
    # p=0.3, b=1.5: negative edge → clamped to exactly 0, never negative
    assert half_kelly(0.3, 1.5) == 0.0


def test_half_kelly_bounded_to_half():
    # Even a near-certain win never exceeds the Half-Kelly ceiling of 0.5.
    for p in (0.6, 0.75, 0.9, 0.99, 1.0):
        f = half_kelly(p, 1.5)
        assert 0.0 <= f <= 0.5


def test_half_kelly_monotonic_in_probability():
    ps = [0.45, 0.5, 0.6, 0.7, 0.8]
    vals = [half_kelly(p, 1.5) for p in ps]
    assert vals == sorted(vals)
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_half_kelly_zero_ratio_is_safe():
    # Degenerate avg_win_loss_ratio must not divide by zero.
    assert half_kelly(0.7, 0.0) == 0.0


def test_kelly_strength_is_half_kelly_normalized():
    # strength = half_kelly / HALF_KELLY_MAX — a true [0, 1] rescale, not the
    # old min(half_kelly, 1.0) that left the raw fraction capped at 0.5.
    for p in (0.45, 0.5, 0.6, 0.75):
        assert kelly_strength(p, 1.5) == pytest.approx(half_kelly(p, 1.5) / HALF_KELLY_MAX)


def test_kelly_strength_reaches_full_size_at_certainty():
    # p=1.0 → half_kelly=0.5=HALF_KELLY_MAX → strength=1.0 (cap now reachable,
    # so the sizer's [0, 1] clamp is a live safety bound, not dead code).
    assert kelly_strength(1.0, 1.5) == pytest.approx(1.0)


def test_kelly_strength_bounded_to_unit_interval():
    for p in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        s = kelly_strength(p, 1.5)
        assert 0.0 <= s <= 1.0


def test_kelly_strength_zero_at_or_below_breakeven():
    # No edge → no size; never negative.
    assert kelly_strength(0.4, 1.5) == pytest.approx(0.0, abs=1e-12)
    assert kelly_strength(0.3, 1.5) == 0.0


def test_kelly_strength_monotonic_in_probability():
    ps = [0.45, 0.5, 0.6, 0.7, 0.8]
    vals = [kelly_strength(p, 1.5) for p in ps]
    assert vals == sorted(vals)


def test_kelly_strength_roughly_doubles_realistic_signal():
    # The whole point: a realistic calibrated prob that previously sized at the
    # raw Half-Kelly fraction now sizes at ~2x (1 / HALF_KELLY_MAX = 2.0).
    raw = half_kelly(0.55, 1.5)
    assert kelly_strength(0.55, 1.5) == pytest.approx(raw * 2.0)
