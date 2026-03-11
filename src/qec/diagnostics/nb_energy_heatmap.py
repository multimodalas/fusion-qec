"""
v8.1.0 — NB Energy Heatmap.

Projects the squared NB eigenvector onto variable and check nodes
to produce per-node energy heatmaps.  Returns normalized heat values
suitable for visualization and repair targeting.

Complements the v7.7.0 ``spectral_heatmaps`` module by providing a
lightweight interface focused on energy distribution only.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics._spectral_utils import build_directed_edges
from src.qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum


_ROUND = 12


def compute_nb_energy_heatmap(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute NB eigenvector energy heatmap over Tanner graph nodes.

    For each directed edge (u, v) with energy |v_e|^2, the energy
    is distributed to source node u.  Variable-node and check-node
    heats are then normalized independently.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``variable_heat`` : list[float] — heat per variable node
        - ``check_heat`` : list[float] — heat per check node
        - ``max_variable_heat`` : float
        - ``max_check_heat`` : float
        - ``num_variable_nodes`` : int
        - ``num_check_nodes`` : int
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    nb = compute_nb_spectrum(H_arr)
    edge_energy = nb["edge_energy"]

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    # Accumulate energy onto source nodes
    var_heat = np.zeros(n, dtype=np.float64)
    chk_heat = np.zeros(m, dtype=np.float64)

    for idx, (u, _v) in enumerate(directed_edges):
        energy = edge_energy[idx]
        if u < n:
            var_heat[u] += energy
        else:
            chk_heat[u - n] += energy

    # Normalize each to [0, 1]
    max_var = float(np.max(var_heat)) if n > 0 else 0.0
    max_chk = float(np.max(chk_heat)) if m > 0 else 0.0

    if max_var > 0:
        var_heat = var_heat / max_var
    if max_chk > 0:
        chk_heat = chk_heat / max_chk

    return {
        "variable_heat": [round(float(v), _ROUND) for v in var_heat],
        "check_heat": [round(float(c), _ROUND) for c in chk_heat],
        "max_variable_heat": round(max_var, _ROUND),
        "max_check_heat": round(max_chk, _ROUND),
        "num_variable_nodes": n,
        "num_check_nodes": m,
    }
