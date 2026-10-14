import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_n_risk_balance.py"
SPEC = importlib.util.spec_from_file_location("p3_n_risk_balance", SCRIPT)
risk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(risk)


def test_balance_and_map_certificate_are_exact():
    rng = np.random.default_rng(19)
    cost = rng.random((40, 31))
    weights = (.2, .65, .1, .05)
    _, trace = risk.dose.bayes_trace(cost, weights)
    truth = np.linspace(0, 25, len(trace))
    rows, _, _ = risk.balance(trace, truth, .1, weights)
    assert max(row["balance_residual"] for row in rows) <= 1e-14
    assert max(row["cumulative_residual"] for row in rows) <= 1e-13
    assert not any(row["map_error"] > .5 and row["certificate_margin"] < -1e-14
                   for row in rows)


def test_flat_observation_has_zero_correction_for_fixed_truth():
    weights = (.2, .65, .1, .05)
    cost = np.ones((12, 20))
    _, trace = risk.dose.bayes_trace(cost, weights)
    rows, _, _ = risk.balance(trace, np.zeros(len(trace)), .1, weights)
    assert max(abs(row["observation_correction"]) for row in rows) <= 1e-14
