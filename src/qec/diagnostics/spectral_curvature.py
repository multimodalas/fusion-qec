"""
v8.1.0 — NB Spectral Curvature Diagnostic.

Estimates the sensitivity of the dominant NB eigenvalue to small
graph perturbations using incremental NB updates.

High curvature means the spectral radius changes significantly
under small edge perturbations, indicating structural fragility.

Uses sparse operators only — no dense NB matrix construction.
Uses incremental warm-started eigenpair updates from v7.9.0.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum
from src.qec.diagnostics.spectral_repair import (
    _undirected_edges_from_H,
    apply_repair_candidate,
    _validate_candidate,
)
from src.qec.diagnostics.spectral_incremental import (
    update_nb_eigenpair_incremental,
)


_ROUND = 12


def estimate_nb_spectral_curvature(
    H: np.ndarray,
    *,
    max_probes: int = 10,
) -> dict[str, Any]:
    """Estimate NB spectral curvature via edge perturbation probing.

    Generates degree-preserving edge swap probes and measures how
    much the spectral radius changes.  The curvature is the mean
    absolute change across probes.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    max_probes : int
        Maximum number of perturbation probes to evaluate.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``mean_curvature`` : float — mean |delta_lambda| across probes
        - ``max_curvature`` : float — max |delta_lambda|
        - ``num_probes`` : int — number of probes evaluated
        - ``baseline_spectral_radius`` : float — unperturbed spectral radius
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    # Compute baseline spectrum
    baseline = compute_nb_spectrum(H_arr)
    baseline_radius = baseline["spectral_radius"]
    baseline_eigvec = baseline["eigenvector"]

    # Generate deterministic edge swap probes
    undirected_edges = _undirected_edges_from_H(H_arr)
    num_edges = len(undirected_edges)

    if num_edges < 2:
        return {
            "mean_curvature": 0.0,
            "max_curvature": 0.0,
            "num_probes": 0,
            "baseline_spectral_radius": round(baseline_radius, _ROUND),
        }

    deltas = []
    probes_done = 0

    # Deterministic probe generation: pair consecutive edges
    for i in range(num_edges):
        if probes_done >= max_probes:
            break

        for j in range(i + 1, num_edges):
            if probes_done >= max_probes:
                break

            edge1 = undirected_edges[i]
            edge2 = undirected_edges[j]

            # Build swap candidate
            var_a, chk_b = edge1[0], edge1[1]
            var_c, chk_d = edge2[0], edge2[1]

            if var_a == var_c or chk_b == chk_d:
                continue

            row_b = chk_b - n
            row_d = chk_d - n

            candidate = {
                "edge1": [row_b, var_a],
                "edge2": [row_d, var_c],
                "new_edge1": [row_d, var_a],
                "new_edge2": [row_b, var_c],
            }

            if not _validate_candidate(H_arr, candidate):
                continue

            # Apply swap and compute perturbed spectral radius
            H_perturbed = apply_repair_candidate(H_arr, candidate)

            incr = update_nb_eigenpair_incremental(
                H_perturbed, baseline_eigvec,
                max_iter=20, tol=1e-8,
            )

            if not incr["converged"]:
                # Fallback to full computation
                perturbed = compute_nb_spectrum(H_perturbed)
                perturbed_radius = perturbed["spectral_radius"]
            else:
                perturbed_radius = incr["spectral_radius"]

            delta = abs(perturbed_radius - baseline_radius)
            deltas.append(delta)
            probes_done += 1

    if len(deltas) == 0:
        return {
            "mean_curvature": 0.0,
            "max_curvature": 0.0,
            "num_probes": 0,
            "baseline_spectral_radius": round(baseline_radius, _ROUND),
        }

    mean_curv = float(np.mean(deltas))
    max_curv = float(np.max(deltas))

    return {
        "mean_curvature": round(mean_curv, _ROUND),
        "max_curvature": round(max_curv, _ROUND),
        "num_probes": len(deltas),
        "baseline_spectral_radius": round(baseline_radius, _ROUND),
    }
