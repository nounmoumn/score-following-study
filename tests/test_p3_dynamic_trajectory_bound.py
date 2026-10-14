import numpy as np

from scripts import p3_dc_conditional_erasure as dc
from scripts import p3_dynamic_trajectory_bound as bound


def test_zero_action_dynamic_upper_is_only_roundoff():
    weights=(0.2,0.65,0.1,0.05);trace=[np.full(20,1/20)]
    for _ in range(29):trace.append(dc.apply_prediction(trace[-1],weights))
    result=bound.dynamic_upper_bound(np.asarray(trace),25,0.1,weights,0.1)
    assert abs(result["trajectory_upper"]) < 1e-12


def test_correction_is_zero_for_zero_delta():
    assert bound.correction_at(np.zeros(5),2.3,0.1)==0.0
