import numpy as np

from scripts import p3_sky_piano_position_transfer as transfer


def test_median_state_uses_left_quantile():
    assert transfer.median_state(np.asarray([0.2, 0.3, 0.5])) == 1
    assert transfer.median_state(np.asarray([0.49, 0.01, 0.5])) == 1


def test_frozen_transfer_constants_match_batik_protocol():
    assert transfer.WINDOW_SEC == 12.0
    assert transfer.HORIZON_SEC == 0.5
    assert transfer.RADIUS == 0.5211881394162992
    assert transfer.LANDMARK_STEP_SEC == 5.0
