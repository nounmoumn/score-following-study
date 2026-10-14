import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p3_prime_effects_and_gates_recompute():
    result = json.loads((ROOT / "results/p3_prime_recovery/result.json").read_text(encoding="utf-8"))
    works = result["works_rows"]
    assert result["performances"] == 40 and result["works"] == len(works) == 23
    for field in ("erasure_distance", "nonlocal_switch_increase", "natural_recovery_burden",
                  "observation_rescue_improvement", "restart_rescue_improvement",
                  "combined_rescue_improvement"):
        assert math.isclose(sum(r[field] for r in works) / len(works), result["metrics"][field]["mean"])
    gates = result["component_gates"]
    assert gates == {"removal": True, "nonlocal_switch": False,
                     "combined_rescue": False, "observation": False, "restart": False}
    assert result["gate_pass"] is False and result["decision"] == "partial_or_failed"


def test_rescue_conditions_share_the_erased_history():
    result = json.loads((ROOT / "results/p3_prime_recovery/result.json").read_text(encoding="utf-8"))
    assert all(r["pre_max_distance"] == 0 for r in result["rows"])
    assert result["metrics"]["erasure_distance"]["ci95"][0] > 0
    assert result["metrics"]["combined_rescue_improvement"]["ci95"][1] < 0

