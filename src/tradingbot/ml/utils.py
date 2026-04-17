"""ML utility functions."""

from __future__ import annotations


def half_kelly(p: float, avg_win_loss_ratio: float = 1.5) -> float:
    """Half-Kelly criterion for position sizing.

    Args:
        p: Predicted win probability from model.
        avg_win_loss_ratio: Historical avg_win / avg_loss.
            Default 1.5 based on backtest: 1h avg=1.52, 4h avg=2.07
            across BTC/ETH/SOL. Conservative estimate.

    Returns:
        Fraction of capital to risk (0.0–1.0).
    """
    q = 1.0 - p
    b = avg_win_loss_ratio
    full_kelly = (p * b - q) / b if b > 0 else 0.0
    return max(0.0, full_kelly * 0.5)
