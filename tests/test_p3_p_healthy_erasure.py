import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_p_healthy_erasure.py"
SPEC = importlib.util.spec_from_file_location("p3_p_healthy_erasure", SCRIPT)
erasure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(erasure)


def test_expected_risk_zero_at_truth_and_positive_away():
    trace = np.zeros((2, 5))
    trace[0, 2] = 1
    trace[1, 4] = 1
    result = erasure.expected_risk(trace, np.array([2., 2.]), .1)
    np.testing.assert_allclose(result, [0, .2])


def test_aggregate_uses_equal_family_means():
    rows = [{"family": "a", "risk_effect": 1., "map_effect": 2., "erased_event": 1, "intact_event": 0},
            {"family": "a", "risk_effect": 3., "map_effect": 4., "erased_event": 0, "intact_event": 0},
            {"family": "b", "risk_effect": 5., "map_effect": 6., "erased_event": 1, "intact_event": 1}]
    values = {row["family"]: row for row in erasure.aggregate(rows)}
    assert values["a"]["risk_effect"] == 2
    assert values["a"]["incidence_effect"] == .5
