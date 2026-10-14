import numpy as np

from score_following.drift import (
    largest_drift_episode,
    window_mean_percentile,
    before_after_window_test,
    circular_shift_null,
    cross_correlation_by_lag,
    drift_rate,
    peak_lag,
    smooth,
    upward_crossings,
)


def test_smooth_preserves_length_and_flattens_spike():
    x = np.zeros(50)
    x[25] = 10.0
    s = smooth(x, 5)
    assert len(s) == len(x)
    assert s[25] < x[25]
    assert np.isclose(s.sum(), x.sum(), rtol=0.3)  # 질량은 대충 보존


def test_drift_rate_positive_on_growing_error_zero_on_shrinking():
    hop = 0.1
    growing = np.linspace(0, 5, 100)     # 오차가 꾸준히 커짐 = 드리프트
    shrinking = np.linspace(5, 0, 100)   # 오차가 줄어듦 = 회복
    dr_grow = drift_rate(growing, hop, smooth_frames=5, rectify=True)
    dr_shrink = drift_rate(shrinking, hop, smooth_frames=5, rectify=True)
    assert dr_grow.mean() > 0.4          # (5초/10초) ≈ 0.5초/초
    assert dr_shrink.max() == 0.0        # 정류했으므로 회복 구간은 전부 0


def test_cross_correlation_finds_known_positive_lag():
    """b가 a를 LAG 프레임 뒤늦게 따라가면, 교차상관 피크가 +LAG에 나와야 한다.
    (우리 규약: 양의 lag = a가 먼저 온다)"""
    rng = np.random.default_rng(0)
    n, LAG = 400, 12
    a = smooth(rng.normal(size=n), 9)      # 자기상관 있는 매끄러운 신호
    b = np.roll(a, LAG)                     # a를 LAG만큼 지연시킨 것
    lags, corrs = cross_correlation_by_lag(a, b, max_lag_frames=40)
    pl, pc = peak_lag(lags, corrs)
    assert pl == LAG, f"피크 lag가 {pl}, 기대값 {LAG}"
    assert pc > 0.9


def test_cross_correlation_sign_convention_is_not_flipped():
    """방향 규약이 뒤집혀 있지 않은지 — b가 a보다 **먼저** 오면 피크는 음의 lag."""
    rng = np.random.default_rng(1)
    n, LAG = 400, 15
    base = smooth(rng.normal(size=n), 9)
    a = np.roll(base, LAG)   # a가 base보다 늦게 = base가 먼저
    b = base
    lags, corrs = cross_correlation_by_lag(a, b, max_lag_frames=40)
    pl, _ = peak_lag(lags, corrs)
    assert pl == -LAG, f"피크 lag가 {pl}, 기대값 {-LAG} (규약 반전 의심)"


def test_circular_shift_null_destroys_relationship():
    """순환이동 귀무분포의 피크 상관은 진짜 관계의 피크 상관보다 뚜렷히 작아야 한다."""
    rng = np.random.default_rng(2)
    n, LAG = 500, 10
    a = smooth(rng.normal(size=n), 9)
    b = np.roll(a, LAG)
    lags, corrs = cross_correlation_by_lag(a, b, max_lag_frames=40)
    _, real_peak = peak_lag(lags, corrs)
    null_lags, null_corrs = circular_shift_null(a, b, 40, n_shuffles=50, seed=3)
    assert real_peak > np.nanpercentile(null_corrs, 95)


def test_upward_crossings_dedupes_chatter():
    x = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype=float)
    # threshold=0.5 기준 상향교차는 인덱스 2, 5, 11
    assert list(upward_crossings(x, 0.5, min_gap_frames=1)) == [2, 5, 11]
    # 떨림 제거(gap=5)면 2와 5는 하나로 묶여서 2, 11만 남는다
    assert list(upward_crossings(x, 0.5, min_gap_frames=5)) == [2, 11]


def test_before_after_window_test_is_symmetric():
    """event가 trigger 직전에만 있으면 n_before만 잡히고, 직후에만 있으면 n_after만."""
    n_frames = 200
    triggers = np.array([100])
    before_only = np.array([90])
    after_only = np.array([110])
    nb, na, nt = before_after_window_test(before_only, triggers, 20, n_frames)
    assert (nb, na, nt) == (1, 0, 1)
    nb, na, nt = before_after_window_test(after_only, triggers, 20, n_frames)
    assert (nb, na, nt) == (0, 1, 1)


def test_largest_drift_episode_finds_known_rampup():
    """오차가 평평 → 하강 → 큰 상승인 궤적에서, 최대 상승 구간의 시작이
    '하강이 끝난 지점(최저점)'으로 잡혀야 한다."""
    err = np.concatenate([
        np.full(30, 1.0),          # 평평
        np.linspace(1.0, 0.2, 20), # 하강(회복)
        np.linspace(0.2, 6.0, 40), # 큰 상승 = 최대 드리프트 사건
        np.full(10, 6.0),
    ])
    start, end, gain = largest_drift_episode(err)
    assert 48 <= start <= 51, f"시작이 최저점(≈49) 근처여야 함, 실제 {start}"
    assert np.isclose(gain, 5.8, atol=0.1)
    assert end > start


def test_largest_drift_episode_flat_trace_gives_no_episode():
    start, end, gain = largest_drift_episode(np.full(50, 2.0))
    assert gain == 0.0


def test_window_mean_percentile_high_when_preceded_by_peak():
    """center 직전 구간만 값이 높으면 백분위가 높게, 낮으면 낮게 나와야 한다."""
    x = np.zeros(300)
    x[100:120] = 5.0          # center(=120) 직전 20프레임만 높음
    pct_hi = window_mean_percentile(x, 120, 20, side="before")
    pct_lo = window_mean_percentile(x, 60, 20, side="before")
    assert pct_hi > 95, f"직전이 높은데 백분위 {pct_hi}"
    assert pct_lo < 90, f"직전이 평평한데 백분위 {pct_lo}"
