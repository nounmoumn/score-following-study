import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_projected_action_matches_direct_recurrence():
    path = ROOT / "scripts/p3_fp_forward_projection.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rng = np.random.default_rng(4); trace = rng.random((8, 9)); trace /= trace.sum(axis=1, keepdims=True)
    weights = (.2, .65, .1, .05)
    actual = module.projected_action(trace, weights, 2, 6, 2)
    expected = np.zeros(9)
    for frame in range(2, 6):
        prior = module.dc.apply_prediction(trace[frame-1], weights)
        expected = module.dc.apply_prediction(expected, weights) + trace[frame] - prior
    expected = module.dc.apply_prediction(module.dc.apply_prediction(expected, weights), weights)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
