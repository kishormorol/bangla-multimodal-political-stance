"""Scoring behaviour, especially around the degenerate cases this data produces."""

from __future__ import annotations

import numpy as np

from bmpb.metrics import bootstrap_macro_f1, score


def test_perfect_prediction():
    y = [0, 1, 2, 0, 1, 2]
    scores = score(y, y, bootstrap=False)
    assert scores.accuracy == 1.0
    assert scores.macro_f1 == 1.0
    assert scores.confusion == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_single_class_prediction_is_flagged():
    # This is exactly what the published mt5 predictions do: "Neutral" for all 47.
    y_true = [0] * 16 + [1] * 15 + [2] * 16
    y_pred = [1] * 47
    scores = score(y_true, y_pred, bootstrap=False)
    assert any("single class" in note for note in scores.notes)
    assert scores.macro_f1 < 0.2


def test_small_test_set_is_flagged():
    scores = score([0, 1, 2] * 10, [0, 1, 2] * 10, bootstrap=False)
    assert any("not meaningful" in note for note in scores.notes)


def test_majority_baseline_is_reported():
    y_true = [0] * 8 + [1] * 2
    scores = score(y_true, [0] * 10, bootstrap=False)
    assert scores.majority_baseline_accuracy == 0.8


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 60)
    y_pred = y_true.copy()
    y_pred[:12] = (y_pred[:12] + 1) % 3
    scores = score(y_true, y_pred)
    low, high = scores.macro_f1_ci95
    assert low <= scores.macro_f1 <= high
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_is_seeded():
    y_true = [0, 1, 2] * 12
    y_pred = [0, 1, 1] * 12
    assert bootstrap_macro_f1(np.array(y_true), np.array(y_pred), rounds=200, seed=1) == (
        bootstrap_macro_f1(np.array(y_true), np.array(y_pred), rounds=200, seed=1)
    )
