#!/usr/bin/env python3
"""Deterministic sonification experiment — v72.5.1.

Generates a small fixed dataset, runs the batch pipeline,
prints a summary, and writes sonification_results.json.
"""

import importlib.util
import json
import os
import sys

_src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _src)


def _load(name, path):
    """Load a module by file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the three sonification modules directly (avoids broken __init__.py)
_exp = os.path.join(_src, "qec", "experiments")
_load("qec.experiments.sonification_interpretation",
      os.path.join(_exp, "sonification_interpretation.py"))
_load("qec.experiments.sonification_comparison",
      os.path.join(_exp, "sonification_comparison.py"))
_batch = _load("qec.experiments.sonification_batch",
               os.path.join(_exp, "sonification_batch.py"))
run_sonification_batch = _batch.run_sonification_batch

# Step 1 — fixed deterministic dataset
results = [
    {"columns": [0, 1, 0, 1, 0, 1], "errorRate": 0.05,
     "complexity": 3.2, "invariants": [[0.0, 0.2], [0.5, 0.7]]},
    {"columns": [1, 1, 0, 0, 1, 1], "errorRate": 0.10,
     "complexity": 4.1, "invariants": [[0.1, 0.3], [0.6, 0.8]]},
    {"columns": [0, 0, 1, 1, 1, 0], "errorRate": 0.03,
     "complexity": 2.5, "invariants": [[0.0, 0.15], [0.4, 0.6]]},
]

# Step 2 — run batch
summary = run_sonification_batch(results)

# Step 3 — print summary
print(f"Samples: {summary['n_samples']}")
print(f"Mean Score: {summary['mean_score']:.2f}")
print("Verdicts:")
for verdict, count in sorted(summary["verdict_counts"].items()):
    print(f"  {verdict}: {count}")
print(f"Best: {summary['best_index']}")
print(f"Worst: {summary['worst_index']}")

# Step 4 — write JSON artifact
with open("sonification_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nWrote sonification_results.json")
