"""Tests for ML diagnostics helpers — calibration, distribution, top features."""

from __future__ import annotations

import math

import numpy as np

from tradingbot.ml.diagnostics import (
    brier_score,
    calibration_error,
    evaluate_calibration,
    summarize_distribution,
    top_features,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_proba) == 0.0

    def test_worst_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_proba) == 1.0

    def test_random_predictions(self):
        # All 0.5 → MSE = 0.25 regardless of labels
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.full(6, 0.5)
        assert brier_score(y_true, y_proba) == 0.25

    def test_empty_returns_nan(self):
        assert math.isnan(brier_score(np.array([]), np.array([])))


class TestCalibrationError:
    def test_perfectly_calibrated(self):
        # 100 samples, half labeled 1, all predicted 0.5 → confidence == accuracy
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.full(100, 0.5)
        ece, mce = calibration_error(y_true, y_proba, n_bins=10)
        assert ece == 0.0
        assert mce == 0.0

    def test_overconfident_model(self):
        # Predicts 1.0 always but truth is 50% — bin 9 has gap 0.5
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.full(100, 0.99)
        ece, mce = calibration_error(y_true, y_proba, n_bins=10)
        # ECE ≈ MCE ≈ 0.49 (entire mass in last bin, gap = |0.5 - 0.99|)
        assert 0.45 <= ece <= 0.5
        assert 0.45 <= mce <= 0.5

    def test_empty_returns_nan(self):
        ece, mce = calibration_error(np.array([]), np.array([]))
        assert math.isnan(ece)
        assert math.isnan(mce)


class TestEvaluateCalibration:
    def test_full_metrics_with_calibrator(self):
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.integers(0, 2, n).astype(float)
        # Raw: noisy probabilities; calibrated: closer to truth
        raw = np.clip(y_true * 0.5 + rng.normal(0.4, 0.1, n), 0, 1)
        calibrated = np.clip(y_true * 0.7 + rng.normal(0.15, 0.05, n), 0, 1)

        m = evaluate_calibration(y_true, raw, calibrated, n_bins=10)
        assert m.n_samples == n
        assert 0.0 <= m.positive_rate <= 1.0
        assert m.brier_calibrated < m.brier_raw, "Calibrated Brier should improve"
        assert not math.isnan(m.ece_raw)
        assert not math.isnan(m.ece_calibrated)
        assert not math.isnan(m.mce_calibrated)

    def test_no_calibrator_keeps_calibrated_nan(self):
        y_true = np.array([0, 1, 0, 1])
        raw = np.array([0.2, 0.8, 0.3, 0.7])
        m = evaluate_calibration(y_true, raw, calibrated_proba=None)
        assert not math.isnan(m.brier_raw)
        assert math.isnan(m.brier_calibrated)
        assert math.isnan(m.ece_calibrated)

    def test_empty_returns_default(self):
        m = evaluate_calibration(np.array([]), np.array([]))
        assert m.n_samples == 0
        assert math.isnan(m.brier_raw)


class TestSummarizeDistribution:
    def test_basic_stats(self):
        probs = np.linspace(0.0, 1.0, 101)  # uniform 0..1
        s = summarize_distribution(probs, thresholds=(0.30, 0.45, 0.50))
        assert s.n == 101
        assert s.min == 0.0
        assert s.max == 1.0
        assert abs(s.p50 - 0.5) < 1e-9
        # 0.30 ≤ p covers 71 values (0.30 .. 1.00 inclusive = 71 of 101)
        assert s.above["0.30"] == 71
        assert s.above_pct["0.30"] == 71 / 101

    def test_handles_nans(self):
        probs = np.array([0.1, np.nan, 0.5, 0.9, np.nan])
        s = summarize_distribution(probs, thresholds=(0.45,))
        assert s.n == 3
        assert s.above["0.45"] == 2

    def test_empty_zero_counts(self):
        s = summarize_distribution(np.array([]), thresholds=(0.45,))
        assert s.n == 0
        assert s.above["0.45"] == 0
        assert s.above_pct["0.45"] == 0.0
        assert math.isnan(s.min)


class TestTopFeatures:
    def test_returns_top_n_sorted_desc(self):
        importance = {"a": 1.0, "b": 5.0, "c": 3.0, "d": 7.0}
        top = top_features(importance, top_n=2)
        assert top == [("d", 7.0), ("b", 5.0)]

    def test_top_n_larger_than_dict(self):
        importance = {"a": 1.0, "b": 2.0}
        top = top_features(importance, top_n=10)
        assert len(top) == 2
        assert top[0] == ("b", 2.0)

    def test_empty_dict(self):
        assert top_features({}, top_n=5) == []
