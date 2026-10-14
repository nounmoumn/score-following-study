import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gt_free_trigger_development_gate_and_frozen_model():
    result = json.loads((ROOT / "results/p3_r_trigger/result.json").read_text())
    metrics = result["metrics"]
    assert result["n"] == 160 and result["events"] == 26 and result["works"] == 23
    assert metrics["auroc"] >= 0.75
    assert metrics["average_precision"] >= 2 * metrics["prevalence"]
    assert metrics["recall_at_threshold"] >= 0.40
    assert metrics["positive_events_at_threshold"] >= 10
    assert result["gate_pass"]
    model = result["frozen_full_development_model"]
    assert len(model["feature_order"]) == len(model["coefficient"]) == 15
    assert model["threshold"] == 0.6906448413477382

