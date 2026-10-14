import numpy as np

from score_following.prospective import (completed_window_values, grouped_predictions,
                                         shift_within_performance, forecast_comparison)
from score_following.evidence_ablation import CostMatrixDTW, synthetic_pair, replay_error
from score_following.online_dtw import OnlineDTW


def test_completed_windows_do_not_read_future_or_interpolate():
    end = np.array([2., 3., 4., 5.])
    a = np.array([.1, .2, .9, .8])
    got = completed_window_values(np.array([1., 2., 3.5]), end, a)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], [.1, .2])
    a[2:] = -999
    np.testing.assert_allclose(completed_window_values(np.array([2., 3.5]), end, a), [.1, .2])


def test_entire_heldout_composition_labels_cannot_change_its_predictions():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 3))
    y = rng.normal(size=60)
    groups = np.repeat(np.arange(3), 20)
    first = grouped_predictions(x, y, groups)
    y[groups == 1] = 1e6
    second = grouped_predictions(x, y, groups)
    np.testing.assert_allclose(first[groups == 1], second[groups == 1])


def test_shift_keeps_performance_and_joint_feature_identity():
    x = np.column_stack([np.arange(20), 2*np.arange(20)])
    perfs = np.repeat([0, 1], 10)
    shifted = shift_within_performance(x, perfs, np.random.default_rng(1))
    np.testing.assert_array_equal(shifted[:, 1], 2*shifted[:, 0])
    assert set(shifted[:10, 0]) == set(range(10))
    assert not np.array_equal(x, shifted)


def test_prospective_model_detects_known_extra_predictive_signal():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(400, 3))
    extra = rng.normal(size=(400, 2))
    y = base[:, 0] + 2*extra[:, 0] + rng.normal(0, .1, 400)
    groups = np.repeat(np.arange(10), 40)
    result, *_ = forecast_comparison(base, extra, y, groups, groups,
                                     n_shifts=19, n_bootstrap=200)
    assert result["bootstrap_ci95"][0] > 1
    assert result["shift_p_upper"] == .05


def test_cost_matrix_matches_original_tracker_and_erasure_preserves_row_mean():
    rng = np.random.default_rng(5)
    u, v = rng.random((30, 12)), rng.random((35, 12))
    original = OnlineDTW(u, v, band_width=5)
    fast = CostMatrixDTW(u, v, band_width=5)
    for i in range(len(u)):
        for j in range(len(v)):
            assert abs(original._cost(i, j) - fast._cost(i, j)) < 1e-12
    assert original.run() == fast.run()
    erased = CostMatrixDTW(u, v, band_width=5, erase=(10, 20))
    np.testing.assert_allclose(erased.costs.mean(axis=1), fast.costs.mean(axis=1))
    np.testing.assert_allclose(np.ptp(erased.costs[10:20], axis=1), 0)
    np.testing.assert_array_equal(erased.costs[:10], fast.costs[:10])


def test_erasure_has_no_pre_onset_effect():
    u, v, truth = synthetic_pair(10, .2, frames=120, onset=40)
    intact = replay_error(u, v, truth, hop=.1, band=10)
    erased = replay_error(u, v, truth, hop=.1, band=10, erase=(40, 70))
    np.testing.assert_array_equal(intact[:40], erased[:40])


def test_early_stop_is_identical_to_original_real_path():
    rng = np.random.default_rng(19)
    u, v = rng.random((25, 12)), rng.random((100, 12))
    full = OnlineDTW(u, v, band_width=5)
    early = OnlineDTW(u, v, band_width=5)
    full.run()
    early.run(stop_at_performance_end=True)
    assert early.path == full.real_path()
