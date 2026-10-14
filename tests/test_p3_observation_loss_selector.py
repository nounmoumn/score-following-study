import numpy as np

from scripts import p3_observation_loss_selector as selector


def test_event_features_are_finite_and_past_only_shape():
    trace = np.full((150, 20), 1 / 20.0)
    features = selector.event_features(trace, 130, 0.1, (0.2, 0.6, 0.2), 0.1)
    assert set(features) == {
        "help_lower", "help_upper", "help_deficit", "posterior_entropy",
        "negative_pmax", "negative_action_tv",
    }
    assert all(np.isfinite(value) for value in features.values())


def test_loto_midpoint_perfect_separation():
    rows = [
        {"track": "a", "x": 2.0, "observation_loss": True},
        {"track": "a", "x": 0.0, "observation_loss": False},
        {"track": "b", "x": 3.0, "observation_loss": True},
        {"track": "b", "x": -1.0, "observation_loss": False},
    ]
    assert selector.loto_midpoint(rows, "x")["balanced_accuracy"] == 1.0
