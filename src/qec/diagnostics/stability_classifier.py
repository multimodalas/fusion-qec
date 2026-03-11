"""
v8.1.0 — Ternary Stability Classifier.

Predicts BP decoding stability directly from spectral invariants
of the Tanner graph structure using simple deterministic threshold
logic.

Output classes:
    +1  →  stable BP regime
     0  →  metastable / oscillatory regime
    -1  →  unstable decoding regime

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics.compute_spectral_metrics import compute_spectral_metrics


_ROUND = 12

# ── Heuristic threshold constants ────────────────────────────────
#
# These are initial values tuned on small Tanner graphs.
# They should be refined with empirical calibration data.

_ENTROPY_HIGH = 2.0        # Above this: well-delocalized (stable)
_ENTROPY_LOW = 0.5         # Below this: strongly localized (unstable)
_SPECTRAL_RADIUS_CRIT = 3.0  # Below this with high entropy: stable
_SIS_HIGH = 0.15           # Above this: unstable
_SPECTRAL_GAP_LOW = 0.1    # Below this: near-degenerate (metastable)
_BETHE_MARGIN_NEG = 0.0    # Below this: structurally unstable


def classify_tanner_graph_stability(
    metrics: dict[str, float],
) -> int:
    """Classify Tanner graph BP stability from spectral invariants.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing spectral metrics.  Expected keys:

        - ``spectral_radius`` : float
        - ``entropy`` : float (spectral mode entropy)
        - ``sis`` : float (spectral instability score)
        - ``spectral_gap`` : float (NB spectral gap)
        - ``bethe_margin`` : float (smallest Bethe Hessian eigenvalue)

    Returns
    -------
    int
        Stability class:
        +1 = stable, 0 = metastable, -1 = unstable.
    """
    entropy = float(metrics.get("entropy", 0.0))
    spectral_radius = float(metrics.get("spectral_radius", 0.0))
    sis = float(metrics.get("sis", 0.0))
    spectral_gap = float(metrics.get("spectral_gap", 0.0))
    bethe_margin = float(metrics.get("bethe_margin", 0.0))

    # Rule 1: Strong instability indicators → unstable
    if sis > _SIS_HIGH or entropy < _ENTROPY_LOW:
        return -1

    # Rule 2: Negative Bethe margin → unstable
    if bethe_margin < _BETHE_MARGIN_NEG:
        return -1

    # Rule 3: Stable regime — high entropy, moderate spectral radius
    if entropy > _ENTROPY_HIGH and spectral_radius < _SPECTRAL_RADIUS_CRIT:
        return +1

    # Rule 4: Stable with good spectral gap and positive margin
    if spectral_gap > _SPECTRAL_GAP_LOW and bethe_margin > 0.0:
        if sis < _SIS_HIGH * 0.5:
            return +1

    # Rule 5: Default → metastable
    return 0


def classify_from_parity_check(H: np.ndarray) -> dict[str, Any]:
    """Compute spectral metrics and classify stability from H.

    Convenience function that computes all required spectral
    invariants and runs the ternary classifier.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``stability_class`` : int (+1, 0, or -1)
        - ``stability_label`` : str ("stable", "metastable", "unstable")
        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``sis`` : float
        - ``spectral_gap`` : float
        - ``bethe_margin`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Compute all spectral diagnostics via aggregator
    metrics = compute_spectral_metrics(H_arr)

    stability_class = classify_tanner_graph_stability(metrics)

    label_map = {+1: "stable", 0: "metastable", -1: "unstable"}

    return {
        "stability_class": stability_class,
        "stability_label": label_map[stability_class],
        "spectral_radius": metrics["spectral_radius"],
        "entropy": metrics["entropy"],
        "sis": metrics["sis"],
        "spectral_gap": metrics["spectral_gap"],
        "bethe_margin": metrics["bethe_margin"],
    }
