import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_o_balance_transfer.py"
SPEC = importlib.util.spec_from_file_location("p3_o_balance_transfer", SCRIPT)
balance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(balance)


def test_auc_and_observation_increment_direction():
    rows = []
    for family in ("a", "b", "c", "d"):
        for index in range(6):
            event = index >= 4
            transport = float(index)
            correction = -float(index)
            rows.append({"family": family, "event": event,
                         "transport_burden": transport,
                         "balance_pressure": transport - correction})
    balance_auc, transport_auc, delta = balance.aucs(rows)
    assert balance_auc == 1.0
    assert transport_auc == 1.0
    assert delta == 0.0


def test_shift_returns_finite_auc_and_preserves_range():
    rows = [{"family": "a", "event": index > 2, "transport_burden": index,
             "balance_pressure": float(index)} for index in range(6)]
    assert np.isfinite(balance.shifted_auc(rows, np.random.default_rng(3)))

