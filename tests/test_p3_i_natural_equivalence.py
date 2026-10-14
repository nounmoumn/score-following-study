import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_i_natural_equivalence.py"
SPEC = importlib.util.spec_from_file_location("p3_i_natural_equivalence", SCRIPT)
natural = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(natural)


def test_flat_likelihood_has_zero_information_and_contrast_has_more():
    flat = np.ones((3, 20))
    contrast = flat.copy()
    contrast[:, 0] = 0
    contrast[:, 1:] = 1
    assert np.allclose(natural.likelihood_information(flat), 0)
    assert np.all(natural.likelihood_information(contrast) > 0)


def test_replay_matches_full_trace_segment():
    rng = np.random.default_rng(4)
    cost = rng.random((40, 25))
    transition = (0.2, 0.65, 0.1, 0.05)
    _, trace = natural.dose.bayes_trace(cost, transition)
    segment = natural.replay(trace[9], cost[10:25], transition)
    assert np.allclose(segment, trace[10:25])

