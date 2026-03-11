"""
v8.1.0 — Graph Repair Suggestions.

Extends the v7.8.0 spectral repair framework to use the instability
score for prioritizing edge swaps that reduce overall instability.

Uses eigenvector energy and localization to propose edge swaps
that reduce the composite instability score.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.diagnostics.spectral_entropy import compute_spectral_mode_entropy
from src.qec.diagnostics.instability_score import compute_instability_score
from src.qec.diagnostics.spectral_repair import (
    propose_repair_candidates,
    apply_repair_candidate,
)


_ROUND = 12


def _compute_metrics_for_H(H: np.ndarray) -> dict[str, float]:
    """Compute the spectral metrics needed for instability scoring."""
    nb = compute_nb_spectrum(H)
    entropy = compute_spectral_mode_entropy(nb["eigenvector"])
    return {
        "sis": nb["sis"],
        "entropy": entropy,
        "spectral_radius": nb["spectral_radius"],
    }


def suggest_graph_repairs(
    H: np.ndarray,
    diagnostics: dict[str, Any] | None = None,
    *,
    top_k_edges: int = 10,
    max_candidates: int = 20,
) -> dict[str, Any]:
    """Suggest graph repairs that reduce the instability score.

    Generates edge swap candidates using the v7.8.0 repair framework,
    then scores them by instability score reduction.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    diagnostics : dict or None
        Pre-computed diagnostics (ignored if None; recomputed).
    top_k_edges : int
        Number of top hot edges to use as repair anchors.
    max_candidates : int
        Maximum number of candidates to evaluate.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``baseline_score`` : float — instability score before repair
        - ``suggestions`` : list[dict] — ranked repair suggestions
        - ``num_evaluated`` : int — number of candidates evaluated
        - ``best_improvement`` : float — best score reduction found
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Baseline instability score
    baseline_metrics = _compute_metrics_for_H(H_arr)
    baseline_result = compute_instability_score(baseline_metrics)
    baseline_score = baseline_result["instability_score"]

    # Generate repair candidates
    candidates = propose_repair_candidates(
        H_arr,
        top_k_edges=top_k_edges,
        max_candidates=max_candidates,
    )

    suggestions = []
    for candidate in candidates:
        H_repaired = apply_repair_candidate(H_arr, candidate)

        repaired_metrics = _compute_metrics_for_H(H_repaired)
        repaired_result = compute_instability_score(repaired_metrics)
        repaired_score = repaired_result["instability_score"]

        delta = repaired_score - baseline_score

        suggestions.append({
            "candidate": candidate,
            "baseline_score": round(baseline_score, _ROUND),
            "repaired_score": round(repaired_score, _ROUND),
            "delta_score": round(delta, _ROUND),
        })

    # Sort: best improvement first (most negative delta)
    # Deterministic tie-breaking by candidate edge tuples
    suggestions.sort(key=lambda s: (
        s["delta_score"],
        tuple(s["candidate"]["edge1"]),
        tuple(s["candidate"]["edge2"]),
    ))

    best_improvement = 0.0
    if suggestions and suggestions[0]["delta_score"] < 0:
        best_improvement = abs(suggestions[0]["delta_score"])

    return {
        "baseline_score": round(baseline_score, _ROUND),
        "suggestions": suggestions,
        "num_evaluated": len(suggestions),
        "best_improvement": round(best_improvement, _ROUND),
    }
