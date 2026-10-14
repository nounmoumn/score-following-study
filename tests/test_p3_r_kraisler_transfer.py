import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kraisler_transfer_negative_result_is_preserved():
    result = json.loads((ROOT / "results/p3_r_kraisler_transfer/result.json").read_text())
    assert result["coverage"] == {"eligible_tracks": 13, "used_tracks": 13, "interventions": 50}
    assert result["incidence"]["events"] == 7
    assert result["incidence"]["affected_tracks"] == 4
    assert result["trigger"]["predicted_positive"] == 0
    assert result["metrics"]["selector_improvement"]["mean"] == 0.0
    assert result["gates"]["coverage"]
    assert result["gates"]["oracle"]
    assert not result["gates"]["incidence"]
    assert not result["gates"]["trigger"]
    assert not result["gates"]["selector"]
    assert not result["gates"]["safety"]
    assert not result["gates"]["all"]
    assert result["decision"] == "stop_before_preserved_holdout"

