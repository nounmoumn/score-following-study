# -*- coding: utf-8 -*-
"""
P3 — "모호성 진입 → 위치 상실 → 드리프트"의 **시간 순서** 검정.
설계 출발점: FEEDBACK.md "P3 설계 (모호성→위치상실→드리프트의 시간순서)" 절(전략, 2026-09-08).

⚠️ 전략 설계안을 그대로 구현하지 않고 두 가지를 고쳤다. 이유를 남겨둔다(LOGGING_PROTOCOL 4):

**(1) "모호진입 후 가장 가까운 드리프트시작까지의 lag" 만으로는 검정이 안 된다 — 순환논증.**
앞(미래)만 보고 가장 가까운 이벤트를 찾으면 lag는 **정의상 항상 양수**다. "양수가 나왔다"는
사실 자체가 증거가 될 수 없다. 그래서 여기서는 **대칭 검정**을 쓴다:
  - 연속형(주 분석): `amb(t)`와 `drift_rate(t)`의 **교차상관을 lag -L..+L 전체에서** 계산하고
    **피크가 양의 lag에 있는지**(모호성이 먼저 오는지) 본다. 음의 lag가 더 크면 가설과 반대.
  - 이벤트형(보조, 전략 설계안에 가까움): 드리프트 시작 시점 기준으로 **직전 W초 창**과
    **직후 W초 창**에 모호진입이 있었는지를 각각 세서 비교한다(before vs after). 이건
    앞뒤가 대칭이라 "정의상 양수"가 아니다.

**(2) 귀무가설은 단순 셔플이 아니라 순환이동(circular shift)이어야 한다.**
`amb(t)`(곡 구조라서 매끄럽다)와 `err(t)`(드리프트는 누적이라 매우 매끄럽다) 둘 다
자기상관이 크다. 자기상관 있는 두 시계열을 단순 셔플과 비교하면 유의성이 과대추정된다
(P1에서 "풀링 상관의 곡간 교란"으로 이미 한 번 데인 것과 같은 계열의 함정).
순환이동은 **각 시계열의 자기상관 구조를 그대로 보존**하면서 둘 사이의 시간 관계만 깨므로
이 경우의 올바른 귀무분포다.
"""
import numpy as np

PITCH_CLASSES = 12


def smooth(x, window_frames):
    """중심 이동평균. window_frames가 1 이하면 그대로 반환."""
    x = np.asarray(x, dtype=float)
    if window_frames <= 1:
        return x.copy()
    k = int(window_frames)
    if k % 2 == 0:
        k += 1  # 중심 정렬을 위해 홀수로
    kernel = np.ones(k) / k
    pad = k // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(padded, kernel, mode="valid")


def drift_rate(err, hop_sec, smooth_frames, rectify=True):
    """오차 궤적에서 '드리프트 속도'(초/초)를 뽑는다.

    err을 이동평균으로 매끄럽게 한 뒤 시간 미분. rectify=True면 음수(오차가 줄어드는
    구간 = 회복)를 0으로 깎는다 — P3가 말하는 '드리프트'는 오차의 **누적/증가**이므로.
    rectify=False 버전도 같이 보고해서 정류 때문에 생긴 결과가 아님을 확인할 것.
    """
    e = smooth(err, smooth_frames)
    d = np.gradient(e) / hop_sec
    return np.maximum(d, 0.0) if rectify else d


def cross_correlation_by_lag(a, b, max_lag_frames):
    """lag = -L..+L 각각에서 corr(a[t], b[t+lag])를 계산.

    양의 lag = a가 b보다 **먼저** 온다(a(t)가 미래의 b(t+lag)와 상관).
    즉 P3 가설이 맞으면 amb를 a, drift_rate를 b로 넣었을 때 피크가 양의 lag에 있어야 한다.
    반환: (lags, corrs) — 표본이 부족하거나 분산이 0인 lag는 nan.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    L = int(max_lag_frames)
    lags = np.arange(-L, L + 1)
    corrs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
            continue
        corrs[i] = np.corrcoef(x, y)[0, 1]
    return lags, corrs


def peak_lag(lags, corrs):
    """상관이 가장 큰 lag(프레임)와 그 값. 전부 nan이면 (nan, nan)."""
    if np.all(np.isnan(corrs)):
        return np.nan, np.nan
    idx = int(np.nanargmax(corrs))
    return int(lags[idx]), float(corrs[idx])


def circular_shift_null(a, b, max_lag_frames, n_shuffles=200, seed=42,
                         min_shift_frac=0.1):
    """순환이동 귀무분포: b를 무작위로 순환이동시킨 뒤 같은 피크-lag 통계를 반복 계산.

    각 시계열의 자기상관은 그대로 보존되고 둘 사이의 시간 관계만 깨진다.
    min_shift_frac: 너무 작은 이동(원본과 거의 같은 것)을 배제하는 하한(길이 비율).
    반환: (peak_lags, peak_corrs) 각각 shape (n_shuffles,)
    """
    rng = np.random.default_rng(seed)
    n = len(b)
    lo = max(1, int(n * min_shift_frac))
    hi = max(lo + 1, n - lo)
    peak_lags, peak_corrs = [], []
    for _ in range(n_shuffles):
        shift = int(rng.integers(lo, hi))
        b_shift = np.roll(b, shift)
        lags, corrs = cross_correlation_by_lag(a, b_shift, max_lag_frames)
        pl, pc = peak_lag(lags, corrs)
        peak_lags.append(pl)
        peak_corrs.append(pc)
    return np.array(peak_lags, dtype=float), np.array(peak_corrs, dtype=float)


def upward_crossings(x, threshold, min_gap_frames=1):
    """x가 threshold를 아래→위로 처음 넘는 지점들의 인덱스.
    min_gap_frames 안에 연달아 나오는 것은 하나로 본다(같은 사건의 떨림 제거)."""
    x = np.asarray(x, dtype=float)
    above = x >= threshold
    idx = np.nonzero(above[1:] & ~above[:-1])[0] + 1
    if len(idx) == 0:
        return idx
    kept = [idx[0]]
    for i in idx[1:]:
        if i - kept[-1] >= min_gap_frames:
            kept.append(i)
    return np.array(kept)


def largest_drift_episode(err):
    """오차 궤적에서 **가장 크게 지속 상승한 구간**(=최대 드리프트 사건)의
    (시작 인덱스, 끝 인덱스, 상승량)을 반환. 없으면 (None, None, 0.0).

    "최대 이익 구간" 문제와 같아서 한 번 훑기(O(n))로 구한다: 진행하면서 지금까지의
    최솟값을 기억하고, 현재값 - 최솟값이 최대가 되는 지점을 찾는다.

    왜 이걸 쓰는가(2026-09-17): 임계값 기반 "드리프트 사건" 검출은 추적이 잘 된 곡에서
    수십 ms짜리 잔떨림을 사건으로 잡아버린다(실측: Bach는 오차 중앙값이 31ms인데
    '드리프트 사건'이 16개로 잡혔다). 곡마다 **가장 큰 사건 하나**만 보면 임계값이
    필요 없고, 그게 바로 P3가 말하는 '위치 상실 후 드리프트 누적'에 해당한다.

    ⚠️⚠️ **알려진 선택 편향 — 이 함수로 모호성을 비교하면 안 된다** (2026-09-17 정량 확인):
    반환하는 시작 인덱스는 **오차의 국소 최솟값**이다. 그런데 P1이 "오차와 모호성은 양의
    상관"임을 확증했으므로, **오차가 낮은 순간은 모호성도 낮은 경향**이다. 즉 이 시작점
    직전의 모호성을 재면 구조적으로 낮게 나온다.
    실측으로 확인된 편향 크기: 주 검정 백분위의 50 미달폭이 amb↔err 상관에 정확히 비례했다
    (국소/구조 상관비 0.2002/0.1363 = 1.47 vs 미달폭비 12.2/8.6 = 1.42).
    → **P3 주 검정(사건 직전 모호성 백분위)은 이 편향 때문에 폐기됐다.**
    ENGINEERING_LOG.md "검정 1은 폐기해야 한다" 절 참조.
    이 함수를 드리프트 **규모** 측정(gain)에 쓰는 것은 문제없다. 사건 **시점**을 다른 변수와
    비교하는 데 쓸 때만 위 편향이 문제가 된다 — 그 경우 오차 궤적과 독립적인 기준점
    (예: GAP 임계 통과 시점, 악보상 절 경계)을 쓸 것.
    """
    e = np.asarray(err, dtype=float)
    if len(e) < 3:
        return None, None, 0.0
    best_gain, best = 0.0, (None, None)
    min_idx = 0
    for i in range(1, len(e)):
        if e[i] - e[min_idx] > best_gain:
            best_gain = e[i] - e[min_idx]
            best = (min_idx, i)
        if e[i] < e[min_idx]:
            min_idx = i
    return best[0], best[1], float(best_gain)


def window_mean_percentile(x, center_idx, window_frames, side="before"):
    """`x`에서 center_idx 직전(또는 직후) window_frames 구간의 평균이, 그 곡의
    **모든 같은 길이 창의 평균 분포** 안에서 몇 번째 백분위인지 반환.

    곡마다 모호성의 평균·분산이 달라도 비교 가능하게 만드는 내부 정규화이며,
    귀무가설 아래에서는 0~100 균등분포이므로 여러 곡에 걸쳐 "50보다 큰가"를
    부호검정으로 바로 검정할 수 있다(임계값·척도 선택이 필요 없음).
    """
    x = np.asarray(x, dtype=float)
    n, w = len(x), int(window_frames)
    if w < 1 or n < w + 2 or center_idx is None:
        return np.nan
    if side == "before":
        lo, hi = center_idx - w, center_idx
    else:
        lo, hi = center_idx + 1, center_idx + 1 + w
    if lo < 0 or hi > n or hi <= lo:
        return np.nan
    target = float(np.mean(x[lo:hi]))
    # 모든 같은 길이 창의 평균 분포 (이동평균으로 한 번에)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    all_means = (csum[w:] - csum[:-w]) / w
    return float(np.mean(all_means <= target) * 100)


def before_after_window_test(event_idx, trigger_idx, window_frames, n_frames):
    """대칭 창 검정 — "드리프트 시작(trigger) 직전 W초에 모호진입(event)이 있었나"
    vs "직후 W초에 있었나"를 센다.

    앞뒤가 대칭이므로 "앞만 봐서 양수가 나온다"는 순환논증이 없다.
    반환: (n_before, n_after, n_trigger) — trigger 중 앞/뒤 창에 event가 있었던 개수.
    """
    event_idx = np.asarray(event_idx)
    n_before = n_after = 0
    for tr in trigger_idx:
        lo_b, hi_b = max(0, tr - window_frames), tr
        lo_a, hi_a = tr + 1, min(n_frames, tr + 1 + window_frames)
        if np.any((event_idx >= lo_b) & (event_idx < hi_b)):
            n_before += 1
        if np.any((event_idx >= lo_a) & (event_idx < hi_a)):
            n_after += 1
    return n_before, n_after, len(trigger_idx)
