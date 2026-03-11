"""
v8.1.0 — NB Sign Pattern Detector.

Analyzes the sign structure of the dominant NB eigenvector to detect
frustrated subgraphs.  Sign disagreements along edges indicate
potential trapping-set boundaries where BP messages may oscillate.

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


def detect_nb_sign_patterns(
    H: np.ndarray,
) -> dict[str, Any]:
    """Detect sign-pattern structure in the dominant NB eigenvector.

    For each undirected Tanner graph edge {u, v}, examines the signs
    of the corresponding directed-edge eigenvector components.  An
    edge is "frustrated" if the two directed components (u->v) and
    (v->u) have opposite signs.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``num_frustrated_edges`` : int — edges with sign disagreement
        - ``num_concordant_edges`` : int — edges with sign agreement
        - ``frustration_ratio`` : float — fraction of frustrated edges
        - ``num_undirected_edges`` : int — total undirected edges
    """
    H_arr = np.asarray(H, dtype=np.float64)

    nb = compute_nb_spectrum(H_arr)
    eigenvector = nb["eigenvector"]

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    # Build directed edge index
    edge_index: dict[tuple[int, int], int] = {}
    for idx, (u, v) in enumerate(directed_edges):
        edge_index[(u, v)] = idx

    # Collect undirected edges (each appears as two directed edges)
    undirected_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for u, v in directed_edges:
        canonical = (min(u, v), max(u, v))
        if canonical not in seen:
            seen.add(canonical)
            undirected_edges.append((u, v))

    num_frustrated = 0
    num_concordant = 0

    for u, v in undirected_edges:
        idx_uv = edge_index.get((u, v))
        idx_vu = edge_index.get((v, u))

        if idx_uv is None or idx_vu is None:
            continue

        sign_uv = np.sign(eigenvector[idx_uv].real)
        sign_vu = np.sign(eigenvector[idx_vu].real)

        if sign_uv * sign_vu < 0:
            num_frustrated += 1
        else:
            num_concordant += 1

    num_undirected = len(undirected_edges)
    frustration_ratio = (
        num_frustrated / num_undirected if num_undirected > 0 else 0.0
    )

    return {
        "num_frustrated_edges": num_frustrated,
        "num_concordant_edges": num_concordant,
        "frustration_ratio": round(float(frustration_ratio), _ROUND),
        "num_undirected_edges": num_undirected,
    }
