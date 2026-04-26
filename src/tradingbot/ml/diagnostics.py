"""ML model diagnostics — calibration error, prediction distribution, summary helpers.

Used by the `ml-diagnostics` CLI to produce a comparable per-(symbol, timeframe)
report covering both model quality (AUC, calibration, feature importance) and
strategy quality (walk-forward backtest Sharpe / Return / Trades).

All functions are pure and operate on numpy arrays — they never touch disk
or run training. The CLI command wires them around ``MLWalkForwardTrainer``
and ``MLStrategyWalkForward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for a probability classifier."""

    n_samples: int = 0
    positive_rate: float = 0.0
    brier_raw: float = float("nan")
    brier_calibrated: float = float("nan")
    ece_raw: float = float("nan")
    ece_calibrated: float = float("nan")
    mce_calibrated: float = float("nan")


@dataclass
class DistributionStats:
    """Summary stats for a 1-D probability array."""

    n: int = 0
    min: float = float("nan")
    p25: float = float("nan")
    p50: float = float("nan")
    mean: float = float("nan")
    p75: float = float("nan")
    max: float = float("nan")
    above: dict[str, int] = field(default_factory=dict)
    above_pct: dict[str, float] = field(default_factory=dict)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Lower is better; 0 = perfect; 0.25 = random for balanced binary.
    Returns NaN if inputs are empty.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_proba - y_true) ** 2))


def calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float]:
    """Expected and maximum calibration error using equal-width bins.

    Returns (ECE, MCE). NaN when inputs are empty.

    ECE: weighted average of |bin_accuracy - bin_confidence|.
    MCE: maximum |bin_accuracy - bin_confidence| across non-empty bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_true.size == 0:
        return float("nan"), float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``digitize`` returns 1..n_bins for values in (0, 1]; clip to keep 0
    # mapped to bin 1 instead of 0 so empty bin 0 doesn't trip the indexer.
    bin_idx = np.clip(np.digitize(y_proba, edges[1:-1]), 0, n_bins - 1)

    total = len(y_proba)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        weight = mask.sum()
        if weight == 0:
            continue
        bin_conf = float(y_proba[mask].mean())
        bin_acc = float(y_true[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += gap * (weight / total)
        if gap > mce:
            mce = gap
    return float(ece), float(mce)


def evaluate_calibration(
    y_true: np.ndarray,
    raw_proba: np.ndarray,
    calibrated_proba: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Brier + ECE for raw probabilities and (optionally) calibrated ones.

    ``calibrated_proba`` may be None when no calibrator is available; in that
    case calibrated metrics stay NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    raw_proba = np.asarray(raw_proba, dtype=float)
    metrics = CalibrationMetrics()
    if y_true.size == 0:
        return metrics
    metrics.n_samples = int(y_true.size)
    metrics.positive_rate = float(y_true.mean())
    metrics.brier_raw = brier_score(y_true, raw_proba)
    metrics.ece_raw, _ = calibration_error(y_true, raw_proba, n_bins=n_bins)

    if calibrated_proba is not None and len(calibrated_proba) == len(y_true):
        cal = np.asarray(calibrated_proba, dtype=float)
        metrics.brier_calibrated = brier_score(y_true, cal)
        ece_c, mce_c = calibration_error(y_true, cal, n_bins=n_bins)
        metrics.ece_calibrated = ece_c
        metrics.mce_calibrated = mce_c
    return metrics


def summarize_distribution(
    probs: np.ndarray,
    thresholds: tuple[float, ...] = (0.30, 0.45, 0.50),
) -> DistributionStats:
    """Return distribution stats and per-threshold counts for a probability array.

    Useful for catching collapsed calibrators (max << entry_threshold) and
    monitoring how many holdout candles would actually fire entries.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[~np.isnan(probs)]
    stats = DistributionStats(n=int(probs.size))
    if probs.size == 0:
        for t in thresholds:
            key = f"{t:.2f}"
            stats.above[key] = 0
            stats.above_pct[key] = 0.0
        return stats

    stats.min = float(np.min(probs))
    stats.p25 = float(np.percentile(probs, 25))
    stats.p50 = float(np.percentile(probs, 50))
    stats.mean = float(np.mean(probs))
    stats.p75 = float(np.percentile(probs, 75))
    stats.max = float(np.max(probs))
    for t in thresholds:
        key = f"{t:.2f}"
        n_above = int((probs >= t).sum())
        stats.above[key] = n_above
        stats.above_pct[key] = float(n_above) / probs.size
    return stats


def top_features(
    feature_importance: dict[str, float],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-N (feature, importance) pairs sorted desc.

    ``feature_importance`` is the dict produced by ``MLWalkForwardReport``
    (already sorted by gain descending), but we sort defensively so callers
    that pass an unsorted dict still get correct output.
    """
    items = sorted(feature_importance.items(), key=lambda kv: -float(kv[1]))
    return [(name, float(imp)) for name, imp in items[:top_n]]
