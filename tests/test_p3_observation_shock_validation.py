from scripts import p3_observation_shock_validation as validation


def test_frozen_threshold_metrics():
    rows = [
        {"observation_loss": True, "action_tv": validation.THRESHOLD + 1},
        {"observation_loss": True, "action_tv": validation.THRESHOLD - 1},
        {"observation_loss": False, "action_tv": validation.THRESHOLD - 1},
        {"observation_loss": False, "action_tv": validation.THRESHOLD + 1},
    ]
    metrics = validation.fixed_threshold_metrics(rows)
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["confusion"] == {"tp": 1, "fn": 1, "tn": 1, "fp": 1}
