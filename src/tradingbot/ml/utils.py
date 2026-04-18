"""ML utility functions."""

from __future__ import annotations

# Production Kelly fraction for LGBM sizing. 0.5 = Half-Kelly (conservative),
# 0.75 = Three-Quarter Kelly (current — higher returns, higher drawdown).
KELLY_FRACTION = 0.75


def kelly_size(
    p: float,
    avg_win_loss_ratio: float = 1.5,
    fraction: float = 0.5,
) -> float:
    """Fractional-Kelly criterion for position sizing.

    Args:
        p: Predicted win probability from model.
        avg_win_loss_ratio: Historical avg_win / avg_loss.
            Default 1.5 based on backtest: 1h avg=1.52, 4h avg=2.07
            across BTC/ETH/SOL. Conservative estimate.
        fraction: Kelly fraction in [0.0, 1.0].
            0.5 = Half-Kelly (default, conservative),
            0.75 = Three-Quarter Kelly (more aggressive),
            1.0 = Full Kelly (theoretically optimal but blow-up risk).

    Returns:
        Fraction of capital to risk (0.0–1.0).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0.0, 1.0], got {fraction}")
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b if b > 0 else 0.0
    return max(0.0, full_kelly * fraction)


def half_kelly(
    p: float,
    avg_win_loss_ratio: float = 1.5,
    fraction: float = 0.5,
) -> float:
    """Backward-compat alias for ``kelly_size``. Prefer ``kelly_size`` directly."""
    return kelly_size(p, avg_win_loss_ratio, fraction)
