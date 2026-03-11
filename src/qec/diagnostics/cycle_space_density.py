"""
v8.1.0 — Cycle Space Density Diagnostic.

Estimates the density of short cycles in the Tanner graph that
contribute to BP instability.  Short cycles (especially 4-cycles
and 6-cycles) cause BP message correlations that degrade
convergence.

The cycle space density is computed from the Tanner graph structure
using trace-based counting of short cycles.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12


def compute_cycle_space_density(H: np.ndarray) -> dict[str, Any]:
    """Estimate cycle space density for a Tanner graph.

    Counts short cycles (length 4 and 6) in the Tanner graph
    using trace-based methods on the bipartite adjacency matrix.

    For a bipartite adjacency matrix A:
    - Tr(A^4) counts closed walks of length 4 (includes 4-cycles)
    - Tr(A^6) counts closed walks of length 6 (includes 6-cycles)

    These are normalized by the number of edges to give density.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``num_edges`` : int — number of edges in Tanner graph
        - ``trace_A4`` : float — Tr(A^4), counting 4-walks
        - ``trace_A6`` : float — Tr(A^6), counting 6-walks
        - ``density_4`` : float — Tr(A^4) / num_edges (4-cycle density)
        - ``density_6`` : float — Tr(A^6) / num_edges (6-cycle density)
        - ``cycle_density`` : float — combined short cycle density
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    num_edges = int(np.sum(H_arr != 0))

    if num_edges == 0:
        return {
            "num_edges": 0,
            "trace_A4": 0.0,
            "trace_A6": 0.0,
            "density_4": 0.0,
            "density_6": 0.0,
            "cycle_density": 0.0,
        }

    # Use H^T H and H H^T for efficient trace computation.
    # For bipartite adjacency A = [[0, H^T],[H, 0]]:
    #   A^2 = [[H^T H, 0], [0, H H^T]]
    #   Tr(A^2) = Tr(H^T H) + Tr(H H^T) = 2 * Tr(H^T H)
    #   A^4 = [[( H^T H )^2, 0], [0, (H H^T)^2]]
    #   Tr(A^4) = Tr((H^T H)^2) + Tr((H H^T)^2)
    #   A^6 similar

    HTH = H_arr.T @ H_arr  # (n x n)
    HHT = H_arr @ H_arr.T  # (m x m)

    # Tr(A^4) = Tr((HTH)^2) + Tr((HHT)^2)
    HTH2 = HTH @ HTH
    HHT2 = HHT @ HHT
    trace_A4 = float(np.trace(HTH2) + np.trace(HHT2))

    # Tr(A^6) = Tr((HTH)^3) + Tr((HHT)^3)
    HTH3 = HTH2 @ HTH
    HHT3 = HHT2 @ HHT
    trace_A6 = float(np.trace(HTH3) + np.trace(HHT3))

    density_4 = trace_A4 / num_edges
    density_6 = trace_A6 / num_edges

    # Combined density: weighted sum favoring shorter cycles
    cycle_density = (2.0 * density_4 + density_6) / 3.0

    return {
        "num_edges": num_edges,
        "trace_A4": round(trace_A4, _ROUND),
        "trace_A6": round(trace_A6, _ROUND),
        "density_4": round(density_4, _ROUND),
        "density_6": round(density_6, _ROUND),
        "cycle_density": round(cycle_density, _ROUND),
    }
