"""
v8.1.0 — Composite Instability Score Diagnostic.

Combines multiple spectral metrics into a single normalized
instability score.  Higher values indicate greater BP instability.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import math

import numpy as np


_ROUND = 12

# ── Weight constants ─────────────────────────────────────────────
#
# Weights for combining spectral signals into a single score.
# These are initial heuristic values.

_W_SIS = 0.4              # spectral instability score weight
_W_LOCALIZATION = 0.3     # localization (1 - normalized entropy) weight
_W_SPECTRAL_RADIUS = 0.3  # normalized spectral radius weight


def compute_instability_score(
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Combine spectral metrics into a single instability score.

    The score is a weighted combination:

        score = w1 * SIS + w2 * localization + w3 * normalized_radius

    where:
    - localization = 1 / (1 + entropy)  (high when entropy is low)
    - normalized_radius = tanh(spectral_radius / 5)  (bounded to [0, 1])

    The result is normalized to [0, 1].

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary with spectral metric values.  Expected keys:

        - ``sis`` : float (spectral instability score)
        - ``entropy`` : float (spectral mode entropy)
        - ``spectral_radius`` : float

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``instability_score`` : float in [0, 1]
        - ``sis_contribution`` : float
        - ``localization_contribution`` : float
        - ``radius_contribution`` : float
    """
    sis = float(metrics.get("sis", 0.0))
    entropy = float(metrics.get("entropy", 0.0))
    spectral_radius = float(metrics.get("spectral_radius", 0.0))

    # Normalize SIS: use sigmoid-like mapping to [0, 1]
    sis_norm = math.tanh(sis * 5.0)

    # Localization: high when entropy is low
    localization = 1.0 / (1.0 + entropy)

    # Normalized spectral radius: bounded to [0, 1]
    radius_norm = math.tanh(max(0.0, spectral_radius) / 5.0)

    # Weighted combination
    raw_score = (
        _W_SIS * sis_norm
        + _W_LOCALIZATION * localization
        + _W_SPECTRAL_RADIUS * radius_norm
    )

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, raw_score))

    return {
        "instability_score": round(score, _ROUND),
        "sis_contribution": round(_W_SIS * sis_norm, _ROUND),
        "localization_contribution": round(_W_LOCALIZATION * localization, _ROUND),
        "radius_contribution": round(_W_SPECTRAL_RADIUS * radius_norm, _ROUND),
    }
