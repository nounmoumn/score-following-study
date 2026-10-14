import numpy as np

from scripts.p3_natural_alignment_calibration import align, synthetic_performance


def _events(count):
    return [{"pitches":[40+i%50,52+i%50],"time_sec":float(i),"latest_end_sec":float(i)+.1} for i in range(count)]


def test_alignment_recovers_identity():
    events=_events(40)
    np.testing.assert_array_equal(align(events,events),np.arange(40))


def test_synthetic_truth_and_alignment_are_well_formed():
    score=_events(80);performance,truth=synthetic_performance(score,7);mapping=align(score,performance)
    assert len(mapping)==len(performance)==len(truth)
    assert np.mean(np.abs(mapping[truth>=0]-truth[truth>=0])<=2)>.9
