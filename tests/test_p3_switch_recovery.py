import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_switch_recovery_result_is_finite_and_recomputes():
    result = json.loads((ROOT / "results/p3_switch_recovery/result.json").read_text(encoding="utf-8"))
    rows = result["rows"]
    assert len(rows) == result["switches"] == 56
    assert all(math.isfinite(r["C_pre"]) and math.isfinite(r["switch_LR"]) for r in rows)
    assert sum(r["persistent"] for r in rows) == result["persistent_recovery_failures"] == 11
    assert sum(r["new_failure"] for r in rows) == result["new_failure_switches"] == 22
    assert Counter(r["kind"] for r in rows) == Counter(result["kind_counts"])
    assert Counter(r["kind"] for r in rows if r["persistent"]) == Counter(result["persistent_kind_counts"])
    observation = [r["persistent"] for r in rows if r["kind"] == "observation_led"]
    others = [r["persistent"] for r in rows if r["kind"] != "observation_led"]
    difference = sum(observation) / len(observation) - sum(others) / len(others)
    assert math.isclose(difference, result["observation_led_minus_others"])
    assert result["decision"] == "mixed_or_below_gate"
    assert result["advance_to_smd"] is False

