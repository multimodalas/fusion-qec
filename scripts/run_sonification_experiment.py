#!/usr/bin/env python3
"""Deterministic sonification experiment — v72.5.1.

Generates a small fixed dataset, runs the batch pipeline,
prints a summary, and writes sonification_results.json.
"""

import json
import os
import sys

# Ensure repo root is on path (required for src.* transitive imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qec.experiments.sonification_batch import run_sonification_batch

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
