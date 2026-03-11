"""
v8.1.0 — Bethe Hessian Stability Margin Diagnostic.

Computes the smallest eigenvalue of the Bethe Hessian matrix,
which approximates the BP stability margin.

A positive margin indicates stable BP; a negative margin signals
structural instability where BP is unlikely to converge.

Reuses the existing Bethe Hessian construction from v6.0.0.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.diagnostics.bethe_hessian import compute_bethe_hessian


_ROUND = 12


def compute_bethe_hessian_margin(
    H: np.ndarray,
    r: float | None = None,
) -> dict[str, Any]:
    """Compute the Bethe Hessian stability margin.

    The margin is the smallest eigenvalue of the Bethe Hessian.
    Positive means BP is in a stable regime; negative means
    structural instability is present.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    r : float or None
        Regularization parameter for Bethe Hessian.
        If None, uses the automatic default from ``compute_bethe_hessian``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:

        - ``bethe_margin`` : float — smallest eigenvalue
        - ``margin_positive`` : bool — True if margin > 0 (stable)
        - ``num_negative`` : int — count of negative eigenvalues
        - ``r_used`` : float — regularization parameter used
    """
    result = compute_bethe_hessian(H, r=r)

    margin = result["min_eigenvalue"]

    return {
        "bethe_margin": round(float(margin), _ROUND),
        "margin_positive": margin > 0.0,
        "num_negative": result["num_negative"],
        "r_used": round(result["r_used"], _ROUND),
    }
