import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multisite_negative_result_is_preserved():
    result = json.loads((ROOT / "results/p3_r_multisite/result.json").read_text())
    assert result["protocol"]["interventions"] == 160
    assert result["protocol"]["candidate_k"] == 10
    assert result["event_incidence"]["persistent_events"] == 26
    assert result["event_incidence"]["affected_works"] == 12
    assert result["gates"]["event_incidence"]
    assert not result["gates"]["oracle"]
    assert not result["gates"]["selector"]
    assert not result["gates"]["intact_safety"]
    assert not result["gates"]["all"]

