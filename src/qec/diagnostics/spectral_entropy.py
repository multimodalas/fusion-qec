"""
v8.1.0 — Spectral Mode Entropy Diagnostic.

Computes the Shannon entropy of the squared eigenvector components,
measuring eigenvector localization.  A low entropy indicates strong
localization (energy concentrated on few edges); a high entropy
indicates delocalized structure.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np


_ROUND = 12


def compute_spectral_mode_entropy(v: np.ndarray) -> float:
    """Compute Shannon entropy of the squared eigenvector components.

    Interprets |v_i|^2 / sum(|v|^2) as a probability distribution
    and computes:

        H = -sum(p_i * log(p_i))

    where p_i = |v_i|^2 / sum(|v|^2) and 0*log(0) = 0 by convention.

    Parameters
    ----------
    v : np.ndarray
        Eigenvector (real or complex).

    Returns
    -------
    float
        Shannon entropy of the squared eigenvector distribution.
        Returns 0.0 for zero-length or all-zero vectors.
    """
    v = np.asarray(v, dtype=np.float64)

    if len(v) == 0:
        return 0.0

    sq = np.abs(v) ** 2
    total = sq.sum()

    if total == 0.0:
        return 0.0

    p = sq / total

    # Compute entropy: -sum(p_i * log(p_i)), treating 0*log(0) = 0
    mask = p > 0
    entropy = -np.sum(p[mask] * np.log(p[mask]))

    return round(float(entropy), _ROUND)
