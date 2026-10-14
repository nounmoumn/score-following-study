import numpy as np
import pytest
from score_following.interval_gt import error_bounds, sustained_bounds

def test_bounds_inside_outside_and_point():
    lo,hi=error_bounds([2,0,4],[1,1,4],[3,3,4])
    np.testing.assert_equal(lo,[0,1,0]);np.testing.assert_equal(hi,[1,3,0])

def test_threshold_is_strict_and_confirmation_support():
    e=np.zeros(20);e[9:14]=.5
    assert not sustained_bounds(e,e,5,10)['possible_event']
    e[9:14]=.51
    r=sustained_bounds(e,e,5,10)
    assert r['definite_event'] and r['earliest_possible_onset']==9
    with pytest.raises(ValueError):sustained_bounds(e,e,15,18)

def test_uncertainty_and_first_onset_interval():
    lo=np.zeros(20);hi=np.zeros(20);hi[5:15]=1;lo[8:13]=1
    r=sustained_bounds(lo,hi,5,11)
    assert r['earliest_possible_onset']==5 and r['latest_guaranteed_first_onset']==8

def test_invalid_unknown_rejected():
    for low,high in [(2,1),(np.nan,2)]:
        with pytest.raises(ValueError):error_bounds(0,low,high)

def test_point_equivalence_exhaustive_small_binary_paths():
    for bits in range(256):
        e=np.array([(bits>>i)&1 for i in range(8)])
        expected=any(np.all(e[j:j+3]>.5) for j in range(6))
        r=sustained_bounds(e,e,0,6,dwell=3)
        assert r['definite_event']==r['possible_event']==expected
