"""
Tests for compute_spectral_metrics aggregator (v8.1.0).

Verifies:
  - returned dictionary keys are correct
  - values are numeric
  - function runs deterministically
  - integration with classifier via compute_spectral_metrics
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.diagnostics.compute_spectral_metrics import compute_spectral_metrics
from src.qec.diagnostics.stability_classifier import (
    classify_tanner_graph_stability,
    classify_from_parity_check,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _dense_H():
    """4x6 denser parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
    ], dtype=np.float64)


# ── Key correctness ─────────────────────────────────────────────


_EXPECTED_KEYS = {
    "spectral_radius",
    "entropy",
    "spectral_gap",
    "bethe_margin",
    "support_dimension",
    "curvature",
    "cycle_density",
    "sis",
}


class TestComputeSpectralMetrics:
    """Tests for the compute_spectral_metrics aggregator."""

    def test_returns_correct_keys(self):
        result = compute_spectral_metrics(_small_H())
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_returns_correct_keys_dense(self):
        result = compute_spectral_metrics(_dense_H())
        assert set(result.keys()) == _EXPECTED_KEYS

    def test_values_are_numeric(self):
        result = compute_spectral_metrics(_small_H())
        for key, val in result.items():
            assert isinstance(val, (int, float)), (
                f"{key} is {type(val)}, expected numeric"
            )

    def test_values_are_finite(self):
        result = compute_spectral_metrics(_small_H())
        for key, val in result.items():
            assert np.isfinite(val), f"{key} = {val} is not finite"

    def test_spectral_radius_positive(self):
        result = compute_spectral_metrics(_small_H())
        assert result["spectral_radius"] > 0

    def test_entropy_nonnegative(self):
        result = compute_spectral_metrics(_small_H())
        assert result["entropy"] >= 0.0

    def test_spectral_gap_nonnegative(self):
        result = compute_spectral_metrics(_small_H())
        assert result["spectral_gap"] >= 0.0

    def test_support_dimension_positive(self):
        result = compute_spectral_metrics(_small_H())
        assert result["support_dimension"] > 0.0

    def test_cycle_density_nonnegative(self):
        result = compute_spectral_metrics(_small_H())
        assert result["cycle_density"] >= 0.0

    def test_sis_nonnegative(self):
        result = compute_spectral_metrics(_small_H())
        assert result["sis"] >= 0.0

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_spectral_metrics(H)
        r2 = compute_spectral_metrics(H)
        assert r1 == r2

    def test_deterministic_dense(self):
        H = _dense_H()
        r1 = compute_spectral_metrics(H)
        r2 = compute_spectral_metrics(H)
        assert r1 == r2

    def test_json_serializable(self):
        result = compute_spectral_metrics(_small_H())
        s = json.dumps(result, sort_keys=True)
        parsed = json.loads(s)
        assert set(parsed.keys()) == _EXPECTED_KEYS


# ── Classifier integration ──────────────────────────────────────


class TestClassifierIntegration:
    """Verify classify_from_parity_check uses compute_spectral_metrics."""

    def test_classifier_produces_valid_output(self):
        H = _small_H()
        result = classify_from_parity_check(H)
        assert result["stability_class"] in {-1, 0, +1}

    def test_metrics_feed_classifier(self):
        """compute_spectral_metrics output is accepted by the classifier."""
        metrics = compute_spectral_metrics(_small_H())
        stability = classify_tanner_graph_stability(metrics)
        assert stability in {-1, 0, +1}

    def test_classifier_deterministic_after_refactor(self):
        H = _small_H()
        r1 = classify_from_parity_check(H)
        r2 = classify_from_parity_check(H)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2
