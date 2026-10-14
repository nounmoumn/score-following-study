import numpy as np

from scripts import p3_natural_path_clock_calibration as clock


def groups(pitches):
    return [{"pitches": value, "time_sec": float(i), "latest_end_sec": float(i) + .1} for i, value in enumerate(pitches)]


def test_identity_path_clock():
    sequence = groups([{60, 64}, {62}, {67, 71}, {69}])
    assert np.array_equal(clock.path_clock(sequence, sequence), np.arange(4))


def test_split_chord_still_maps_every_performance_event_monotonically():
    score = groups([{60, 64, 67}, {62, 65, 69}, {64, 67, 71}])
    performance = groups([{60}, {64, 67}, {62, 65}, {69}, {64, 67, 71}])
    mapping = clock.path_clock(score, performance)
    assert len(mapping) == len(performance)
    assert np.all(mapping >= 0)
    assert np.all(np.diff(mapping) >= 0)
    assert mapping[0] == 0 and mapping[-1] == 2
