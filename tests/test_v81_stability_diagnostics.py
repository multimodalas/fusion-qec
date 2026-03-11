"""
Tests for v8.1.0 spectral diagnostics and ternary stability classifier.

Verifies:
  - spectral mode entropy computation
  - NB spectral gap correctness
  - Bethe Hessian margin sign
  - effective support dimension
  - cycle space density
  - stability classifier output stability
  - instability score normalization
  - repair suggestions
  - determinism across runs
  - integration with classify_from_parity_check

Does not require running the full BP decoder.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.diagnostics.spectral_entropy import compute_spectral_mode_entropy
from src.qec.diagnostics.nb_spectral_gap import compute_nb_spectral_gap
from src.qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from src.qec.diagnostics.effective_support_dimension import (
    compute_effective_support_dimension,
)
from src.qec.diagnostics.cycle_space_density import compute_cycle_space_density
from src.qec.diagnostics.stability_classifier import (
    classify_tanner_graph_stability,
    classify_from_parity_check,
)
from src.qec.diagnostics.instability_score import compute_instability_score
from src.qec.diagnostics.repair_suggestions import suggest_graph_repairs


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _identity_H():
    """3x3 identity matrix (minimal graph)."""
    return np.eye(3, dtype=np.float64)


def _dense_H():
    """4x6 denser parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
    ], dtype=np.float64)


# ── Spectral Entropy Tests ──────────────────────────────────────


class TestSpectralEntropy:
    """Tests for compute_spectral_mode_entropy."""

    def test_uniform_vector_max_entropy(self):
        """Uniform vector should have maximum entropy = log(n)."""
        n = 16
        v = np.ones(n) / np.sqrt(n)
        entropy = compute_spectral_mode_entropy(v)
        expected = math.log(n)
        assert abs(entropy - expected) < 1e-8

    def test_localized_vector_zero_entropy(self):
        """Single-entry vector should have entropy = 0."""
        v = np.zeros(10)
        v[3] = 1.0
        entropy = compute_spectral_mode_entropy(v)
        assert entropy == 0.0

    def test_empty_vector(self):
        entropy = compute_spectral_mode_entropy(np.array([]))
        assert entropy == 0.0

    def test_zero_vector(self):
        entropy = compute_spectral_mode_entropy(np.zeros(5))
        assert entropy == 0.0

    def test_entropy_nonnegative(self):
        v = np.array([1.0, 0.5, 0.3, 0.1])
        entropy = compute_spectral_mode_entropy(v)
        assert entropy >= 0.0

    def test_deterministic(self):
        v = np.array([0.5, 0.3, 0.1, 0.8, 0.2])
        e1 = compute_spectral_mode_entropy(v)
        e2 = compute_spectral_mode_entropy(v)
        assert e1 == e2

    def test_two_equal_entries(self):
        """Two equal entries: entropy = log(2)."""
        v = np.array([1.0, 1.0])
        entropy = compute_spectral_mode_entropy(v)
        assert abs(entropy - math.log(2)) < 1e-8


# ── NB Spectral Gap Tests ───────────────────────────────────────


class TestNBSpectralGap:
    """Tests for compute_nb_spectral_gap."""

    def test_returns_all_keys(self):
        result = compute_nb_spectral_gap(_small_H())
        assert "lambda_1" in result
        assert "lambda_2" in result
        assert "spectral_gap" in result

    def test_gap_nonnegative(self):
        result = compute_nb_spectral_gap(_small_H())
        assert result["spectral_gap"] >= 0.0

    def test_lambda_1_geq_lambda_2(self):
        result = compute_nb_spectral_gap(_small_H())
        assert result["lambda_1"] >= result["lambda_2"]

    def test_gap_equals_difference(self):
        result = compute_nb_spectral_gap(_small_H())
        expected_gap = result["lambda_1"] - result["lambda_2"]
        assert abs(result["spectral_gap"] - expected_gap) < 1e-8

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_nb_spectral_gap(H)
        r2 = compute_nb_spectral_gap(H)
        assert r1["lambda_1"] == r2["lambda_1"]
        assert r1["lambda_2"] == r2["lambda_2"]
        assert r1["spectral_gap"] == r2["spectral_gap"]

    def test_identity_matrix(self):
        """Identity matrix has minimal graph structure."""
        result = compute_nb_spectral_gap(_identity_H())
        # Identity has no cycles, very simple structure
        assert result["lambda_1"] >= 0.0


# ── Bethe Hessian Margin Tests ──────────────────────────────────


class TestBetheHessianMargin:
    """Tests for compute_bethe_hessian_margin."""

    def test_returns_all_keys(self):
        result = compute_bethe_hessian_margin(_small_H())
        assert "bethe_margin" in result
        assert "margin_positive" in result
        assert "num_negative" in result
        assert "r_used" in result

    def test_margin_is_float(self):
        result = compute_bethe_hessian_margin(_small_H())
        assert isinstance(result["bethe_margin"], float)

    def test_margin_positive_consistency(self):
        result = compute_bethe_hessian_margin(_small_H())
        if result["bethe_margin"] > 0:
            assert result["margin_positive"] is True
        else:
            assert result["margin_positive"] is False

    def test_num_negative_nonneg(self):
        result = compute_bethe_hessian_margin(_small_H())
        assert result["num_negative"] >= 0

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_bethe_hessian_margin(H)
        r2 = compute_bethe_hessian_margin(H)
        assert r1["bethe_margin"] == r2["bethe_margin"]
        assert r1["margin_positive"] == r2["margin_positive"]

    def test_r_used_positive(self):
        result = compute_bethe_hessian_margin(_small_H())
        assert result["r_used"] > 0


# ── Effective Support Dimension Tests ────────────────────────────


class TestEffectiveSupportDimension:
    """Tests for compute_effective_support_dimension."""

    def test_uniform_vector(self):
        """Uniform vector: D_eff = exp(log(n)) = n."""
        n = 16
        v = np.ones(n) / np.sqrt(n)
        d_eff = compute_effective_support_dimension(v)
        assert abs(d_eff - n) < 1e-4

    def test_localized_vector(self):
        """Single-entry: D_eff = exp(0) = 0 (special case)."""
        v = np.zeros(10)
        v[3] = 1.0
        d_eff = compute_effective_support_dimension(v)
        assert d_eff == 0.0

    def test_empty_vector(self):
        d_eff = compute_effective_support_dimension(np.array([]))
        assert d_eff == 0.0

    def test_positive_for_nonzero(self):
        v = np.array([1.0, 0.5, 0.3])
        d_eff = compute_effective_support_dimension(v)
        assert d_eff > 0.0

    def test_deterministic(self):
        v = np.array([0.5, 0.3, 0.1, 0.8])
        d1 = compute_effective_support_dimension(v)
        d2 = compute_effective_support_dimension(v)
        assert d1 == d2


# ── Cycle Space Density Tests ────────────────────────────────────


class TestCycleSpaceDensity:
    """Tests for compute_cycle_space_density."""

    def test_returns_all_keys(self):
        result = compute_cycle_space_density(_small_H())
        expected_keys = {
            "num_edges", "trace_A4", "trace_A6",
            "density_4", "density_6", "cycle_density",
        }
        assert expected_keys == set(result.keys())

    def test_num_edges_correct(self):
        H = _small_H()
        result = compute_cycle_space_density(H)
        expected = int(np.sum(H != 0))
        assert result["num_edges"] == expected

    def test_densities_nonnegative(self):
        result = compute_cycle_space_density(_small_H())
        assert result["density_4"] >= 0.0
        assert result["density_6"] >= 0.0
        assert result["cycle_density"] >= 0.0

    def test_trace_A4_positive_for_connected(self):
        """Connected graph should have nonzero Tr(A^4)."""
        result = compute_cycle_space_density(_small_H())
        assert result["trace_A4"] > 0.0

    def test_identity_matrix(self):
        """Identity has no short cycles."""
        result = compute_cycle_space_density(_identity_H())
        # Identity: each edge is isolated, few short cycles
        assert result["num_edges"] == 3

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_cycle_space_density(H)
        r2 = compute_cycle_space_density(H)
        assert r1 == r2

    def test_empty_matrix(self):
        H = np.zeros((3, 4), dtype=np.float64)
        result = compute_cycle_space_density(H)
        assert result["num_edges"] == 0
        assert result["cycle_density"] == 0.0


# ── Stability Classifier Tests ──────────────────────────────────


class TestStabilityClassifier:
    """Tests for classify_tanner_graph_stability."""

    def test_output_in_valid_range(self):
        metrics = {
            "spectral_radius": 2.0,
            "entropy": 1.5,
            "sis": 0.1,
            "spectral_gap": 0.5,
            "bethe_margin": 1.0,
        }
        result = classify_tanner_graph_stability(metrics)
        assert result in {-1, 0, +1}

    def test_high_sis_is_unstable(self):
        metrics = {
            "spectral_radius": 5.0,
            "entropy": 0.3,
            "sis": 0.5,
            "spectral_gap": 0.01,
            "bethe_margin": -1.0,
        }
        result = classify_tanner_graph_stability(metrics)
        assert result == -1

    def test_stable_regime(self):
        metrics = {
            "spectral_radius": 1.5,
            "entropy": 3.0,
            "sis": 0.01,
            "spectral_gap": 0.5,
            "bethe_margin": 2.0,
        }
        result = classify_tanner_graph_stability(metrics)
        assert result == +1

    def test_negative_bethe_is_unstable(self):
        metrics = {
            "spectral_radius": 2.0,
            "entropy": 1.5,
            "sis": 0.05,
            "spectral_gap": 0.5,
            "bethe_margin": -0.5,
        }
        result = classify_tanner_graph_stability(metrics)
        assert result == -1

    def test_deterministic(self):
        metrics = {
            "spectral_radius": 2.5,
            "entropy": 1.8,
            "sis": 0.1,
            "spectral_gap": 0.3,
            "bethe_margin": 0.5,
        }
        r1 = classify_tanner_graph_stability(metrics)
        r2 = classify_tanner_graph_stability(metrics)
        assert r1 == r2

    def test_classify_from_parity_check(self):
        """Integration test: classify_from_parity_check produces valid output."""
        H = _small_H()
        result = classify_from_parity_check(H)
        assert "stability_class" in result
        assert "stability_label" in result
        assert result["stability_class"] in {-1, 0, +1}
        assert result["stability_label"] in {"stable", "metastable", "unstable"}

    def test_classify_from_parity_check_deterministic(self):
        H = _small_H()
        r1 = classify_from_parity_check(H)
        r2 = classify_from_parity_check(H)
        assert r1["stability_class"] == r2["stability_class"]
        assert r1["spectral_radius"] == r2["spectral_radius"]
        assert r1["entropy"] == r2["entropy"]

    def test_classify_from_parity_check_json_serializable(self):
        H = _small_H()
        result = classify_from_parity_check(H)
        s = json.dumps(result, sort_keys=True)
        parsed = json.loads(s)
        assert parsed["stability_class"] == result["stability_class"]


# ── Instability Score Tests ──────────────────────────────────────


class TestInstabilityScore:
    """Tests for compute_instability_score."""

    def test_returns_all_keys(self):
        metrics = {"sis": 0.1, "entropy": 1.5, "spectral_radius": 2.0}
        result = compute_instability_score(metrics)
        assert "instability_score" in result
        assert "sis_contribution" in result
        assert "localization_contribution" in result
        assert "radius_contribution" in result

    def test_score_in_zero_one(self):
        metrics = {"sis": 0.1, "entropy": 1.5, "spectral_radius": 2.0}
        result = compute_instability_score(metrics)
        assert 0.0 <= result["instability_score"] <= 1.0

    def test_high_sis_high_score(self):
        low = compute_instability_score(
            {"sis": 0.01, "entropy": 3.0, "spectral_radius": 1.0}
        )
        high = compute_instability_score(
            {"sis": 1.0, "entropy": 0.1, "spectral_radius": 10.0}
        )
        assert high["instability_score"] > low["instability_score"]

    def test_deterministic(self):
        metrics = {"sis": 0.1, "entropy": 1.5, "spectral_radius": 2.0}
        r1 = compute_instability_score(metrics)
        r2 = compute_instability_score(metrics)
        assert r1 == r2

    def test_zero_inputs(self):
        metrics = {"sis": 0.0, "entropy": 0.0, "spectral_radius": 0.0}
        result = compute_instability_score(metrics)
        assert 0.0 <= result["instability_score"] <= 1.0


# ── Repair Suggestions Tests ────────────────────────────────────


class TestRepairSuggestions:
    """Tests for suggest_graph_repairs."""

    def test_returns_all_keys(self):
        H = _small_H()
        result = suggest_graph_repairs(H, max_candidates=5)
        assert "baseline_score" in result
        assert "suggestions" in result
        assert "num_evaluated" in result
        assert "best_improvement" in result

    def test_baseline_score_in_range(self):
        H = _small_H()
        result = suggest_graph_repairs(H, max_candidates=5)
        assert 0.0 <= result["baseline_score"] <= 1.0

    def test_suggestions_sorted_by_delta(self):
        H = _dense_H()
        result = suggest_graph_repairs(H, max_candidates=10)
        deltas = [s["delta_score"] for s in result["suggestions"]]
        assert deltas == sorted(deltas)

    def test_deterministic(self):
        H = _small_H()
        r1 = suggest_graph_repairs(H, max_candidates=5)
        r2 = suggest_graph_repairs(H, max_candidates=5)
        assert r1["baseline_score"] == r2["baseline_score"]
        assert r1["num_evaluated"] == r2["num_evaluated"]


# ── Integration Tests ────────────────────────────────────────────


class TestIntegration:
    """Integration tests combining multiple v8.1.0 diagnostics."""

    def test_full_pipeline_small_H(self):
        """Full diagnostic pipeline on a small matrix."""
        H = _small_H()

        from src.qec.diagnostics.spectral_nb import compute_nb_spectrum

        nb = compute_nb_spectrum(H)
        entropy = compute_spectral_mode_entropy(nb["eigenvector"])
        support = compute_effective_support_dimension(nb["eigenvector"])
        gap = compute_nb_spectral_gap(H)
        margin = compute_bethe_hessian_margin(H)
        cycles = compute_cycle_space_density(H)

        metrics = {
            "spectral_radius": nb["spectral_radius"],
            "entropy": entropy,
            "sis": nb["sis"],
            "spectral_gap": gap["spectral_gap"],
            "bethe_margin": margin["bethe_margin"],
        }

        stability = classify_tanner_graph_stability(metrics)
        score = compute_instability_score(metrics)

        assert stability in {-1, 0, +1}
        assert 0.0 <= score["instability_score"] <= 1.0
        assert support >= 0.0
        assert cycles["cycle_density"] >= 0.0

    def test_full_pipeline_deterministic(self):
        """Full pipeline produces identical results on repeated runs."""
        H = _small_H()

        r1 = classify_from_parity_check(H)
        r2 = classify_from_parity_check(H)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_classify_dense_H(self):
        """Classifier works on denser matrices."""
        H = _dense_H()
        result = classify_from_parity_check(H)
        assert result["stability_class"] in {-1, 0, +1}

    def test_all_diagnostics_json_serializable(self):
        """All v8.1.0 diagnostics produce JSON-serializable output."""
        H = _small_H()

        from src.qec.diagnostics.spectral_nb import compute_nb_spectrum

        nb = compute_nb_spectrum(H)
        v = nb["eigenvector"]

        entropy = compute_spectral_mode_entropy(v)
        support = compute_effective_support_dimension(v)
        gap = compute_nb_spectral_gap(H)
        margin = compute_bethe_hessian_margin(H)
        cycles = compute_cycle_space_density(H)
        classifier = classify_from_parity_check(H)
        score = compute_instability_score({
            "sis": nb["sis"],
            "entropy": entropy,
            "spectral_radius": nb["spectral_radius"],
        })

        # All should be JSON-serializable
        for name, obj in [
            ("entropy", entropy),
            ("support", support),
            ("gap", gap),
            ("margin", margin),
            ("cycles", cycles),
            ("classifier", classifier),
            ("score", score),
        ]:
            s = json.dumps(obj, sort_keys=True)
            assert len(s) > 0, f"{name} failed JSON serialization"
