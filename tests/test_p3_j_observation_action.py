import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_j_observation_action.py"
SPEC = importlib.util.spec_from_file_location("p3_j_observation_action", SCRIPT)
action = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(action)


def test_decomposition_is_exact_on_random_natural_costs():
    rng = np.random.default_rng(7)
    initial = rng.random(31)
    initial /= initial.sum()
    rows = action.decompose(initial, rng.random((17, 31)), (.2, .65, .1, .05))
    for row in rows:
        np.testing.assert_allclose(row["intact"] - row["erased"],
                                   row["reconstructed"], atol=1e-14)
        assert abs(row["action"].sum()) <= 1e-14


def test_row_constant_observation_has_zero_action():
    rng = np.random.default_rng(8)
    initial = rng.random(23)
    initial /= initial.sum()
    cost = np.broadcast_to(np.arange(9, dtype=float)[:, None], (9, 23))
    rows = action.decompose(initial, cost, (.2, .65, .1, .05))
    for row in rows:
        np.testing.assert_allclose(row["action"], 0, atol=1e-14)
        np.testing.assert_allclose(row["intact"], row["erased"], atol=1e-14)

