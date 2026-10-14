from scripts import p3_fhvb_long_horizon_validation as validation


def test_signal_projection_is_unchanged_and_only_event_horizon_is_longer():
    assert validation.WINDOW_SEC == 12.0
    assert validation.PROJECTION_HORIZON_SEC == 0.5
    assert validation.LAG_SEC == validation.STEP_SEC == 0.5
    assert validation.EVENT_HORIZON_SEC == 1.5
    assert validation.RADIUS == 0.5211881394162992
