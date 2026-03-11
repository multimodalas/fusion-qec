"""
v8.1.0 — NB Localization Detector.

Detects strongly localized eigenmodes in the non-backtracking spectrum
using IPR thresholds.  Returns a boolean detection flag and the set of
localized edge indices for downstream repair targeting.

Complements the v6.1.0 ``nb_localization`` module by providing a
simplified detection interface suitable for automated pipelines.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics.spectral_nb import compute_nb_spectrum
from src.qec.diagnostics._spectral_utils import compute_ipr


_ROUND = 12

# Default threshold: modes with IPR above this are "localized".
_DEFAULT_IPR_THRESHOLD = 0.15


def detect_nb_localization(
    H: np.ndarray,
    *,
    ipr_threshold: float = _DEFAULT_IPR_THRESHOLD,
) -> dict[str, Any]:
    """Detect whether the dominant NB eigenmode is localized.

    A localized dominant mode concentrates eigenvector energy on a
    small subset of edges, indicating potential trapping-set activity.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    ipr_threshold : float
        IPR values above this indicate localization.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``localized`` : bool — True if dominant mode is localized
        - ``ipr`` : float — IPR of the dominant eigenvector
        - ``localized_edges`` : list[int] — edge indices with above-mean energy
        - ``num_localized_edges`` : int — count of localized edges
    """
    H_arr = np.asarray(H, dtype=np.float64)

    nb = compute_nb_spectrum(H_arr)
    ipr = nb["ipr"]
    edge_energy = nb["edge_energy"]

    num_edges = len(edge_energy)
    localized = bool(ipr > ipr_threshold)

    if num_edges > 0:
        mean_energy = float(np.mean(edge_energy))
        # Deterministic: sorted edge indices with above-mean energy
        localized_edges = sorted(
            int(i) for i in range(num_edges)
            if edge_energy[i] > mean_energy
        )
    else:
        localized_edges = []

    return {
        "localized": localized,
        "ipr": round(float(ipr), _ROUND),
        "localized_edges": localized_edges,
        "num_localized_edges": len(localized_edges),
    }
