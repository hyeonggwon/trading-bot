"""ML utility functions."""

from __future__ import annotations

# Structural ceiling of half_kelly(): full_kelly(p=1.0)=1.0 → half = 0.5.
# Used to normalize the raw Half-Kelly fraction into a [0, 1] confidence.
HALF_KELLY_MAX = 0.5


def half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Half-Kelly criterion for position sizing.

    Args:
        p: Predicted win probability from model.
        avg_win_loss_ratio: Historical avg_win / avg_loss.
            Default 1.5 based on backtest: 1h avg=1.52, 4h avg=2.07
            across BTC/ETH/SOL. Conservative estimate.

    Returns:
        Raw Half-Kelly fraction, bounded to [0.0, HALF_KELLY_MAX (=0.5)].
    """
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b if b > 0 else 0.0
    return max(0.0, full_kelly * 0.5)


def kelly_strength(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Map win probability → Signal.strength in a true [0, 1] range.

    The raw Half-Kelly fraction structurally tops out at HALF_KELLY_MAX (0.5),
    so feeding it directly into a sizer clamped at 1.0 made that clamp dead code
    and systematically under-sized ML entries (realistic calibrated probs
    yielded 0.04–0.20). Normalizing by HALF_KELLY_MAX turns strength into a
    genuine [0, 1] confidence: a near-certain signal reaches full base size, and
    the sizer's [0, 1] clamp becomes a real safety cap.

    Args:
        p: Predicted (calibrated) win probability from model.
        avg_win_loss_ratio: Historical avg_win / avg_loss.

    Returns:
        Position-sizing strength in [0.0, 1.0].
    """
    return min(half_kelly(p, avg_win_loss_ratio) / HALF_KELLY_MAX, 1.0)
