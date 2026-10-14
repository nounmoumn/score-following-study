# -*- coding: utf-8 -*-
"""Ground truth 보간. 원본: scripts/p1_verification.py"""
import numpy as np


def piecewise_linear_gt(perf_beats, score_beats):
    """Morsi & Serra 방식: 비트 단위 정렬을 구간별 선형보간해서
    임의의 성능 시각 -> 스코어 시각 함수로 만든다.
    반환: (gt_fn, gt_lo, gt_hi)"""
    perf_beats = np.array(perf_beats)
    score_beats = np.array(score_beats)
    order = np.argsort(perf_beats)
    perf_beats, score_beats = perf_beats[order], score_beats[order]

    def gt(perf_time):
        return np.interp(perf_time, perf_beats, score_beats)

    return gt, perf_beats.min(), perf_beats.max()
