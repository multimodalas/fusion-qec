"""
v8.1.0 — Effective Support Dimension Diagnostic.

Computes the effective support dimension of an eigenvector:

    D_eff = exp(H)

where H is the spectral mode entropy.  This gives the effective
number of edges carrying significant eigenvector weight.

A low D_eff means energy is concentrated on few edges (localized);
a high D_eff means energy is spread across many edges (delocalized).

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import math

import numpy as np

from src.qec.diagnostics.spectral_entropy import compute_spectral_mode_entropy


_ROUND = 12


def compute_effective_support_dimension(v: np.ndarray) -> float:
    """Compute effective support dimension from eigenvector entropy.

    D_eff = exp(H) where H is the spectral mode entropy.

    Parameters
    ----------
    v : np.ndarray
        Eigenvector (real or complex).

    Returns
    -------
    float
        Effective support dimension.  Returns 0.0 for zero or empty vectors.
    """
    entropy = compute_spectral_mode_entropy(v)

    if entropy == 0.0:
        return 0.0

    return round(math.exp(entropy), _ROUND)
