import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_basin_result_is_preserved():
    result = json.loads((ROOT / "results/p3_r_basin/result.json").read_text())
    assert result["protocol"]["n_performances"] == 40
    assert result["protocol"]["n_works"] == 23
    assert result["protocol"]["candidate_k"] == 5
    assert result["diagnostics"]["safety_violation_rate"] == 0.0
    assert not result["gates"]["oracle"]
    assert not result["gates"]["selector"]
    assert result["gates"]["intact_safety"]
    assert not result["gates"]["all"]
    assert result["decision"] == "branch_resolution_or_initialization_bottleneck"
    rows = result["rows"]
    assert sum(row["natural_burden"] > 0.5 for row in rows) == 2

