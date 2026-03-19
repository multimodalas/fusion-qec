"""Import integrity tests (v72.6.0).

Verifies that key modules import without errors and that
the sonification batch pipeline executes on a minimal input.
"""


def test_import_qec():
    import qec  # noqa: F401


def test_import_qec_experiments():
    import qec.experiments  # noqa: F401


def test_import_sonification_batch():
    from qec.experiments.sonification_batch import run_sonification_batch  # noqa: F401


def test_import_sonification_comparison():
    from qec.experiments.sonification_comparison import run_sonification_comparison  # noqa: F401


def test_import_sonification_interpretation():
    from qec.experiments.sonification_interpretation import interpret_sonification_comparison  # noqa: F401


def test_sonification_batch_executes():
    from qec.experiments.sonification_batch import run_sonification_batch

    results = [
        {"columns": [0, 1, 0, 1, 0, 1], "errorRate": 0.05,
         "complexity": 3.2, "invariants": [[0.0, 0.2], [0.5, 0.7]]},
    ]
    summary = run_sonification_batch(results)
    assert summary["n_samples"] == 1
    assert 0.0 <= summary["mean_score"] <= 1.0
    assert summary["best_index"] == 0
    assert summary["worst_index"] == 0
