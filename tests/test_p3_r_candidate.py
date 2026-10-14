import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_coverage_gate_and_diagnostics():
    result = json.loads((ROOT / "results/p3_r_candidate/result.json").read_text(encoding="utf-8"))
    rows = result["rows"]
    assert len(rows) == 40 and result["works"] == 23
    for k in (1, 3, 5, 10):
        assert result[f"performance_recall_at_{k}"] == sum(r[f"recall_at_{k}"] for r in rows) / len(rows)
    assert result["performance_recall_at_5"] == result["performance_recall_at_10"] == .575
    missed = [r for r in rows if not r["recall_at_5"]]
    assert len(missed) == 17
    assert all(min(c["distance_score_sec"] for c in r["candidates"]) <= 2 for r in missed)
    assert result["gate_pass"] is False and result["decision"] == "candidate_loss_bottleneck"

