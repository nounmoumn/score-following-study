import numpy as np

from score_following.ground_truth import piecewise_linear_gt


def test_interpolates_exact_midpoint():
    perf_beats = [0.0, 1.0, 2.0]
    score_beats = [0.0, 2.0, 4.0]  # 스코어가 연주보다 2배 빠르게 흐름
    gt_fn, lo, hi = piecewise_linear_gt(perf_beats, score_beats)

    assert lo == 0.0 and hi == 2.0
    assert gt_fn(0.5) == 1.0  # 정확한 중간값
    assert gt_fn(1.5) == 3.0


def test_handles_unsorted_input():
    perf_beats = [2.0, 0.0, 1.0]
    score_beats = [4.0, 0.0, 2.0]
    gt_fn, lo, hi = piecewise_linear_gt(perf_beats, score_beats)
    assert np.isclose(gt_fn(0.5), 1.0)
