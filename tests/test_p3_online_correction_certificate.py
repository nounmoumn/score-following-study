import numpy as np

from scripts import p3_online_correction_certificate as certificate
from scripts import p3_dc_conditional_erasure as dc


def test_zero_action_has_zero_certificate():
    weights = (0.0, 1.0, 0.0)
    trace = [np.full(10, 0.1)]
    for _ in range(29):
        trace.append(dc.apply_prediction(trace[-1], weights))
    score = certificate.certificate_score(np.asarray(trace), 25, 0.1, weights, 0.1)
    assert abs(score) < 1e-12
