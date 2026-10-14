import importlib.util
import csv
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_e_erasure_transfer.py"
SPEC = importlib.util.spec_from_file_location("p3_e_transfer", SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def test_interventions_flatten_only_erased_and_preserve_sham_contrasts():
    rng = np.random.default_rng(4)
    base = rng.random((20, 15))
    erased, sham = transfer.interventions(base, 5, 12)
    assert np.array_equal(erased[:5], base[:5])
    assert np.array_equal(sham[:5], base[:5])
    assert np.max(np.ptp(erased[5:12], axis=1)) < 1e-12
    assert np.allclose(np.diff(sham[5:12], axis=1), np.diff(base[5:12], axis=1))


def test_summarize_weights_composers_equally():
    rows = ([{"composer": "a", "late_divergence_sec": 1.0}] * 4 +
            [{"composer": "b", "late_divergence_sec": 3.0}] * 4)
    values, metrics = transfer.summarize(rows, np.random.default_rng(1))
    assert values == {"a": 1.0, "b": 3.0}
    assert metrics["mean"] == 2.0


def test_frozen_result_recomputes_and_all_gates_pass():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/p3_e_erasure_transfer/result.json").read_text(encoding="utf8"))
    with (root / "results/p3_e_erasure_transfer/interventions.csv").open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["composer"], []).append(float(row["late_divergence_sec"]))
    values = np.asarray([np.mean(item) for item in grouped.values()])
    assert len(rows) == 120 and len(values) == 30
    assert np.isclose(values.mean(), result["metrics"]["mean"])
    assert np.isclose(np.median(values), result["metrics"]["median"])
    assert np.isclose(np.mean(values > .1), result["metrics"]["share_gt_0_10"])
    assert all(result["gates"].values())
    assert result["p3_e_cross_corpus_supported"] is True
