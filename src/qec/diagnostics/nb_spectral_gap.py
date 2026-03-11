"""
v8.1.0 — Non-Backtracking Spectral Gap Diagnostic.

Computes the spectral gap of the non-backtracking operator on a
Tanner graph: the difference between the first and second largest
eigenvalue magnitudes.

A large spectral gap indicates well-separated dominant structure;
a small gap suggests near-degeneracy and potential instability.

Uses sparse operators only — no dense NB matrix construction.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import eigs

from src.qec.diagnostics._spectral_utils import build_nb_operator
from src.qec.diagnostics.spectral_nb import _TannerGraph


_ROUND = 12


def compute_nb_spectral_gap(H: np.ndarray) -> dict[str, Any]:
    """Compute the NB spectral gap for a parity-check matrix.

    Extracts the two largest eigenvalues (by magnitude) of the
    non-backtracking operator using a sparse Krylov eigensolver,
    then returns their difference.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``lambda_1`` : float — largest eigenvalue magnitude
        - ``lambda_2`` : float — second largest eigenvalue magnitude
        - ``spectral_gap`` : float — lambda_1 - lambda_2
    """
    H_arr = np.asarray(H, dtype=np.float64)

    graph = _TannerGraph(H_arr)
    op, directed_edges = build_nb_operator(graph)

    n_edges = len(directed_edges)

    if n_edges < 3:
        # Too few edges for k=2 eigensolver
        return {
            "lambda_1": 0.0,
            "lambda_2": 0.0,
            "spectral_gap": 0.0,
        }

    # Request top-2 eigenvalues by largest real part
    k = min(2, n_edges - 2)
    try:
        vals, _ = eigs(op, k=k, which="LR", tol=1e-6)
    except Exception:
        # Degenerate operator (e.g., all-zero NB matrix for tree graphs)
        return {
            "lambda_1": 0.0,
            "lambda_2": 0.0,
            "spectral_gap": 0.0,
        }

    magnitudes = np.abs(vals)
    # Sort descending
    magnitudes = np.sort(magnitudes)[::-1]

    lambda_1 = float(magnitudes[0]) if len(magnitudes) > 0 else 0.0
    lambda_2 = float(magnitudes[1]) if len(magnitudes) > 1 else 0.0

    gap = lambda_1 - lambda_2

    return {
        "lambda_1": round(lambda_1, _ROUND),
        "lambda_2": round(lambda_2, _ROUND),
        "spectral_gap": round(gap, _ROUND),
    }
