import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts/p3_dc_conditional_erasure.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def test_exact_conditional_erasure_identity():
    m = module(); rng = np.random.default_rng(7)
    initial = rng.random(11); initial /= initial.sum()
    result = m.decompose(initial, rng.random((9, 11)), np.linspace(2, 6, 9), .1, (.1, .2, .4, .3))
    assert result["posterior_residual"] <= 1e-12
    assert result["risk_residual"] <= 1e-12
    assert np.sign(result["erasure_effect"]) == np.sign(result["projected_help_sum"])


def test_random_conditional_erasure_audit():
    rows = module().random_audit(trials=20)
    assert max(row["posterior_residual"] for row in rows) <= 1e-12
    assert max(row["risk_residual"] for row in rows) <= 1e-12
