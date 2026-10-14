import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_safeguarded_result_separates_headroom_from_ranking():
    result = json.loads((ROOT / "results/p3_r_safeguarded/result.json").read_text())
    assert result["protocol"]["interventions"] == 160
    assert result["event_incidence"]["persistent_events"] == 26
    assert result["gates"]["oracle"]
    assert not result["gates"]["selector"]
    assert not result["gates"]["intact_safety"]
    assert result["switching"]["selector_count"] == 15
    assert result["switching"]["selector_affected_count"] == 5
    assert result["switching"]["safety_count"] == 15

