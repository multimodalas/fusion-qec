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

Public API is defined in ``api.py`` — add new exports there.
"""

from .api import *  # noqa: F401,F403
