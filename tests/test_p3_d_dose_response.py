import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_d_dose_response.py"
SPEC = importlib.util.spec_from_file_location("p3_d_dose_response", SCRIPT)
dose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dose)


def test_attenuation_scales_position_contrast_exactly():
    rng = np.random.default_rng(8)
    base = rng.random((20, 17))
    for value in dose.LAMBDAS:
        changed = dose.attenuate(base, 4, 15, value)
        assert np.array_equal(changed[:4], base[:4])
        assert dose.contrast_residual(base, changed, 4, 15, value) < 1e-12


def test_trace_matches_production_filter_path():
    rng = np.random.default_rng(9)
    cost = rng.random((35, 31))
    transition = (0.2, 0.65, 0.1, 0.05)
    path, trace = dose.bayes_trace(cost, transition)
    assert np.array_equal(path, dose.bayes_from_cost(cost, transition, dose.BETA, dose.RESTART))
    assert np.allclose(trace.sum(axis=1), 1)


def test_js_is_zero_only_for_identical_rows_in_example():
    left = np.asarray([[0.5, 0.5], [0.8, 0.2]])
    right = np.asarray([[0.5, 0.5], [0.2, 0.8]])
    values = dose.js_rows(left, right)
    assert values[0] == 0
    assert values[1] > 0

