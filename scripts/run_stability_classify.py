#!/usr/bin/env python
"""
v8.1.0 — Stability Classification CLI.

Computes spectral diagnostics for a Tanner graph and runs the
ternary stability classifier.

Usage:
    python scripts/run_stability_classify.py --stability-classify

All computation is deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Ensure imports resolve from repository root.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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
from src.qec.diagnostics.stability_classifier import (
    classify_tanner_graph_stability,
    classify_from_parity_check,
)
from src.qec.diagnostics.instability_score import compute_instability_score


def _demo_H() -> np.ndarray:
    """Small demo parity-check matrix."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def run_stability_classify(
    H: np.ndarray,
) -> dict[str, Any]:
    """Run full spectral diagnostics and stability classification.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix.

    Returns
    -------
    dict
        Full diagnostic results including stability class.
    """
    # Step 1: Core spectral diagnostics
    nb = compute_nb_spectrum(H)
    entropy = compute_spectral_mode_entropy(nb["eigenvector"])
    support_dim = compute_effective_support_dimension(nb["eigenvector"])

    # Step 2: Extended spectral invariants
    gap_result = compute_nb_spectral_gap(H)
    margin_result = compute_bethe_hessian_margin(H)
    cycle_result = compute_cycle_space_density(H)

    # Step 3: Ternary classifier
    metrics = {
        "spectral_radius": nb["spectral_radius"],
        "entropy": entropy,
        "sis": nb["sis"],
        "spectral_gap": gap_result["spectral_gap"],
        "bethe_margin": margin_result["bethe_margin"],
    }
    stability_class = classify_tanner_graph_stability(metrics)

    # Step 4: Instability score
    score_result = compute_instability_score(metrics)

    label_map = {+1: "STABLE", 0: "METASTABLE", -1: "UNSTABLE"}

    return {
        "spectral_radius": nb["spectral_radius"],
        "ipr": nb["ipr"],
        "eeec": nb["eeec"],
        "sis": nb["sis"],
        "entropy": entropy,
        "support_dimension": support_dim,
        "spectral_gap": gap_result["spectral_gap"],
        "lambda_1": gap_result["lambda_1"],
        "lambda_2": gap_result["lambda_2"],
        "bethe_margin": margin_result["bethe_margin"],
        "bethe_margin_positive": margin_result["margin_positive"],
        "cycle_density": cycle_result["cycle_density"],
        "stability_class": stability_class,
        "stability_label": label_map[stability_class],
        "instability_score": score_result["instability_score"],
    }


def print_results(result: dict[str, Any]) -> None:
    """Print formatted stability classification results."""
    print("=" * 60)
    print("v8.1.0 — Ternary Stability Classification")
    print("=" * 60)
    print()
    print("Spectral Diagnostics:")
    print(f"  spectral_radius    = {result['spectral_radius']:.6f}")
    print(f"  IPR                = {result['ipr']:.6f}")
    print(f"  EEEC               = {result['eeec']:.6f}")
    print(f"  SIS                = {result['sis']:.6f}")
    print(f"  entropy            = {result['entropy']:.6f}")
    print(f"  support_dimension  = {result['support_dimension']:.6f}")
    print(f"  spectral_gap       = {result['spectral_gap']:.6f}")
    print(f"  lambda_1           = {result['lambda_1']:.6f}")
    print(f"  lambda_2           = {result['lambda_2']:.6f}")
    print(f"  bethe_margin       = {result['bethe_margin']:.6f}")
    print(f"  bethe_margin_pos   = {result['bethe_margin_positive']}")
    print(f"  cycle_density      = {result['cycle_density']:.6f}")
    print()
    print("Classification:")
    print(f"  stability_class    = {result['stability_class']:+d}")
    print(f"  stability_label    = {result['stability_label']}")
    print(f"  instability_score  = {result['instability_score']:.6f}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="run_stability_classify",
        description="Compute spectral diagnostics and classify Tanner graph stability.",
    )
    parser.add_argument(
        "--stability-classify",
        action="store_true",
        help="Run ternary stability classifier on the demo matrix.",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Path to a .npy file containing the parity-check matrix.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    args = parser.parse_args(argv)

    if not args.stability_classify:
        parser.print_help()
        return 0

    # Load matrix
    if args.matrix is not None:
        H = np.load(args.matrix)
    else:
        H = _demo_H()

    result = run_stability_classify(H)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_results(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
