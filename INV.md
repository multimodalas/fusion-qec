# QSOL QEC — Proven Invariants Registry

This document records formally proven invariants discovered and
exploited within the QEC framework. Append-only.

---

## QSOL-BP-INV-001

**Name:**
URW rho=1.0 Baseline Equivalence

**Statement:**
`bp_decode(H, llr, mode="min_sum_urw", urw_rho=1.0, ...)` produces
bitwise-identical output to `bp_decode(H, llr, mode="min_sum", ...)`.

**Scope:**
All schedules (flooding, layered), with or without damping/clipping.

**Proof Summary:**
- Algebraic: URW reweighting multiplies check-to-variable messages by
  rho and divides by rho in the inverse step. At rho=1.0 these cancel
  exactly, leaving the standard min-sum update.
- Execution equivalence: same code path (multiplication by 1.0 is
  identity in float64).

**Validation:**
- `np.testing.assert_array_equal` on correction vectors and iteration
  counts across 15 trials per schedule.
- Tests: `tests/test_urw_bp_v370.py::TestRhoOneInvariance`,
  `tests/test_urw_dps_preview_v370.py::TestGateCheck`.

**Safety:**
- Mutation safety: bp_decode does not mutate inputs.
- Determinism preserved: same seed, same H, same LLR -> same output.
- Ordering unchanged: no loop reordering.

**Exploitation:**
In `test_preview_sweep`, rho=1.0 reuses baseline min_sum benchmark
results instead of running a redundant `run_benchmark` call.

**Impact:**
Eliminates one full benchmark sweep (2 distances x 2 noise levels x 25
trials = 100 decoder calls) per DPS preview run.

**Version Introduced:**
v68.4.1

---

## QSOL-BP-INV-002

**Name:**
Default-Window Sign-Vector Equivalence

**Statement:**
When `tail_window == gos_window == bti_window` (the default: all 12),
the sign vectors `_sign(normed_llr[i])` and CRC32 signatures
`zlib.crc32(_sign(v).astype(int8).tobytes())` computed in
`_compute_msi`, `_compute_cpi`, `_compute_gos`, and `_compute_bti`
are identical pure-function evaluations on identical data
(`normed_llr[-w:]` where `w = min(window, len(trace))`).

**Scope:**
`src/qec/diagnostics/bp_dynamics.py::compute_bp_dynamics_metrics` with
default parameters (all window values equal to 12).

**Proof Summary:**
- Algebraic: `_sign(x) = np.where(x < 0, -1, 1)` is a pure function.
  Same input array produces the same output array, bitwise.
- Execution equivalence: All four metrics compute `_sign()` on the same
  tail slice `normed_llr[-w:]` with identical `w` when window params
  are equal. CRC32 is deterministic over identical byte sequences.

**Validation:**
- `np.testing.assert_array_equal` on precomputed vs inline sign vectors.
- CRC32 integer equality between cached and recomputed signatures.
- Tests: `tests/test_bp_dynamics.py::TestINV002SignCacheEquivalence`.

**Safety:**
- Mutation safety: `_sign()` allocates new arrays; cache entries are
  read-only after construction.
- Determinism preserved: pure functions, no state, no ordering change.
- Ordering unchanged: metric computation order and results identical.

**Exploitation:**
In `compute_bp_dynamics_metrics`, sign vectors and CRC32 signatures
are precomputed once for the shared tail window and passed to MSI, CPI,
GOS, and BTI via optional `_sign_cache` / `_crc32_cache` parameters.
Eliminates ~56 redundant `_sign()` calls per invocation (from 68 down
to 12) for the default window configuration.

**Impact:**
~80% reduction in sign-vector computation within BP dynamics analysis.
Proportional to variable count N per LLR vector. No behavioral change;
all 3783 tests pass identically.

**Version Introduced:**
v68.5.0
