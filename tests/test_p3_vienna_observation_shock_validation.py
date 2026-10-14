from scripts import p3_vienna_observation_shock_validation as validation


def test_frozen_vienna_design():
    assert validation.STEP_SEC == validation.HORIZON_SEC == validation.LAG_SEC == 0.5
    assert validation.WINDOW_SEC == 12.0
    assert validation.PERMUTATIONS == 2_000
