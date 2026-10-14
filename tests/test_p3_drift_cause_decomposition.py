import numpy as np

from scripts import p3_d_dose_response as dose
from scripts import p3_drift_cause_decomposition as decomposition


def test_three_term_decomposition_is_exact():
    rng = np.random.default_rng(3)
    trace = rng.random((20, 13))
    trace /= trace.sum(axis=1, keepdims=True)
    truth = np.linspace(1, 10, len(trace))
    rows = decomposition.decompose(trace, truth, 0.1, (0.1, 0.2, 0.4, 0.3))
    assert max(row["one_step_residual"] for row in rows) < 1e-12
    assert max(row["cumulative_residual"] for row in rows) < 1e-11


def test_event_label_requires_half_positive_burden():
    segment = [
        {"transition_burden": 2.0, "occurrence_burden": 1.0, "observation_correction": -0.5}
    ]
    _, _, label = decomposition.label_event(segment)
    assert label == "transition"
