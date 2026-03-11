"""
v8.1.0 — Aggregated Spectral Metrics.

Single entry point that computes all spectral invariants used by
the ternary stability classifier.  Reuses the NB eigenpair for
entropy and support dimension to avoid redundant computation.

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
from src.qec.diagnostics.nb_spectral_gap import compute_nb_spectral_gap
from src.qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from src.qec.diagnostics.effective_support_dimension import (
    compute_effective_support_dimension,
)
from src.qec.diagnostics.spectral_curvature import estimate_nb_spectral_curvature
from src.qec.diagnostics.cycle_space_density import compute_cycle_space_density


_ROUND = 12


def compute_spectral_metrics(H: np.ndarray) -> dict[str, Any]:
    """Compute all spectral invariants for a parity-check matrix.

    Aggregates the v8.1.0 spectral diagnostics into a single call,
    reusing the NB eigenpair for entropy and support dimension.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``spectral_gap`` : float
        - ``bethe_margin`` : float
        - ``support_dimension`` : float
        - ``curvature`` : float
        - ``cycle_density`` : float
        - ``sis`` : float — spectral instability score (from NB spectrum)
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Core NB spectrum (reused for entropy and support dimension)
    nb = compute_nb_spectrum(H_arr)
    eigenvector = nb["eigenvector"]

    entropy = compute_spectral_mode_entropy(eigenvector)
    support_dimension = compute_effective_support_dimension(eigenvector)

    gap_result = compute_nb_spectral_gap(H_arr)
    margin_result = compute_bethe_hessian_margin(H_arr)
    curvature_result = estimate_nb_spectral_curvature(H_arr)
    cycle_result = compute_cycle_space_density(H_arr)

    return {
        "spectral_radius": round(float(nb["spectral_radius"]), _ROUND),
        "entropy": round(float(entropy), _ROUND),
        "spectral_gap": round(float(gap_result["spectral_gap"]), _ROUND),
        "bethe_margin": round(float(margin_result["bethe_margin"]), _ROUND),
        "support_dimension": round(float(support_dimension), _ROUND),
        "curvature": round(float(curvature_result["mean_curvature"]), _ROUND),
        "cycle_density": round(float(cycle_result["cycle_density"]), _ROUND),
        "sis": round(float(nb["sis"]), _ROUND),
    }
