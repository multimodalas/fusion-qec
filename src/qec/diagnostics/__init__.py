"""
Layer 1a — BP diagnostics (opt-in, additive).

Provides energy-landscape analysis, basin-switching detection,
iteration-trace diagnostics, BP dynamics regime analysis,
regime transition analysis, phase diagram aggregation,
freeze detection, fixed-point trap analysis, basin-of-attraction
analysis, attractor landscape mapping, free-energy barrier
estimation, pseudocodeword boundary estimation,
Tanner spectral fragility diagnostics,
spectral–boundary alignment diagnostics,
spectral trapping-set diagnostics,
BP phase-space exploration,
ternary decoder topology classification,
decoder phase diagram aggregation,
phase boundary analysis,
non-backtracking spectrum diagnostics,
Bethe Hessian spectral analysis,
BP stability proxy metrics,
BP Jacobian spectral radius estimation,
spectral trapping-set candidate detection,
spectral–BP attractor alignment diagnostics,
spectral failure risk scoring,
BP stability prediction,
instability sensitivity maps,
ASCII phase heatmap output,
spectral mode entropy,
NB spectral gap,
Bethe Hessian stability margin,
effective support dimension,
NB spectral curvature,
cycle space density,
ternary stability classification,
composite instability scoring,
graph repair suggestions,
NB localization detection,
NB energy heatmaps,
and NB sign pattern detection
for BP convergence traces.
Does not modify decoder internals.
"""

# ── BP dynamics and convergence diagnostics ─────────────────────

from .basin_probe import probe_local_ternary_basin
from .bethe_hessian import compute_bethe_hessian
from .bethe_hessian_margin import compute_bethe_hessian_margin
from .bp_barrier_analysis import compute_bp_barrier_analysis
from .bp_basin_analysis import compute_bp_basin_analysis
from .bp_boundary_analysis import compute_bp_boundary_analysis
from .bp_dynamics import classify_bp_regime, compute_bp_dynamics_metrics
from .bp_fixed_point_analysis import compute_bp_fixed_point_analysis
from .bp_freeze_detection import compute_bp_freeze_detection
from .bp_jacobian_estimator import estimate_bp_jacobian_spectral_radius
from .bp_landscape_mapping import compute_bp_landscape_map
from .bp_phase_diagram import compute_bp_phase_diagram
from .bp_phase_space import compute_bp_phase_space, compute_metastability_score
from .bp_regime_trace import compute_bp_regime_trace
from .bp_stability_predictor import compute_bp_stability_prediction
from .bp_stability_proxy import estimate_bp_stability

# ── Spectral diagnostics (v6.0–v7.9) ───────────────────────────

from .compute_spectral_metrics import compute_spectral_metrics
from .cycle_space_density import compute_cycle_space_density
from .effective_support_dimension import compute_effective_support_dimension
from .instability_score import compute_instability_score
from .nb_energy_heatmap import compute_nb_energy_heatmap
from .nb_localization import compute_nb_localization_metrics
from .nb_localization_detector import detect_nb_localization
from .nb_sign_pattern_detector import detect_nb_sign_patterns
from .nb_spectral_gap import compute_nb_spectral_gap
from .nb_trapping_candidates import compute_nb_trapping_candidates
from .non_backtracking_spectrum import compute_non_backtracking_spectrum

# ── Phase diagrams and boundaries ───────────────────────────────

from .phase_boundary_analysis import analyze_phase_boundaries
from .phase_diagram import build_decoder_phase_diagram, make_phase_grid
from .phase_heatmap import print_phase_heatmap

# ── Repair and instability mitigation ───────────────────────────

from .repair_suggestions import suggest_graph_repairs
from .sensitivity_map import (
    compute_measured_instability_deltas,
    compute_proxy_sensitivity_scores,
    compute_sensitivity_map,
)
from .spectral_boundary_alignment import compute_spectral_boundary_alignment
from .spectral_bp_alignment import compute_spectral_bp_alignment
from .spectral_curvature import estimate_nb_spectral_curvature
from .spectral_entropy import compute_spectral_mode_entropy
from .spectral_failure_risk import compute_spectral_failure_risk
from .spectral_heatmaps import (
    compute_spectral_heatmaps,
    rank_check_nodes_by_heat,
    rank_edges_by_heat,
    rank_variable_nodes_by_heat,
)
from .spectral_incremental import (
    detect_edge_swap,
    identify_affected_nb_edges,
    score_repair_candidate_incremental,
    update_nb_eigenpair_incremental,
    update_nb_eigenpair_localized,
)
from .spectral_nb import (
    SPECTRAL_SCHEMA_VERSION,
    compute_edge_sensitivity_ranking,
    compute_nb_spectrum,
)
from .spectral_trapping_sets import compute_spectral_trapping_sets

# ── Stability classification (v8.1) ────────────────────────────

from .stability_classifier import (
    classify_from_parity_check,
    classify_tanner_graph_stability,
)
from .tanner_spectral_analysis import compute_tanner_spectral_analysis
from .ternary_decoder_topology import compute_ternary_decoder_topology
