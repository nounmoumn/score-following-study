import numpy as np

from scripts import p3_sky_piano_clock_adapter as adapter


def test_longest_nondecreasing_subsequence_removes_gross_outlier():
    values = np.asarray([0.0, 1.0, 5.0, 2.0, 3.0, 4.0])
    kept = adapter.longest_nondecreasing_indices(values)
    assert np.array_equal(kept, [0, 1, 3, 4, 5])


def test_equal_performance_times_are_collapsed_without_invented_timing():
    score, performance, kept = adapter.adapt_clock(
        np.asarray([0.0, 0.25, 0.5, 0.75]), np.asarray([1.0, 1.125, 1.125, 1.25])
    )
    assert np.array_equal(performance, [1.0, 1.125, 1.25])
    assert np.allclose(score, [0.0, 0.375, 0.75])
    assert len(kept) == 4


def test_adapter_never_moves_surviving_raw_times():
    score = np.asarray([0.0, 1.0, 2.0, 3.0])
    raw = np.asarray([0.0, 4.0, 2.0, 3.0])
    _, performance, _ = adapter.adapt_clock(score, raw)
    assert set(performance).issubset(set(raw))
    assert np.all(np.diff(performance) > 0)
