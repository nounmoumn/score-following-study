from scripts import p3_fhvb_smoothness_prospective as audit


def test_prospective_constants_are_fixed():
    assert audit.CALIBRATION_FRACTION == 0.20
    assert audit.NORMALIZATION_WINDOW_SEC == 12.0
    assert audit.STEP_SEC == 0.5
