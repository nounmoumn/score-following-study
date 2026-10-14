# -*- coding: utf-8 -*-
"""
관계식 도출 — 상관계수(rho)를 넘어서 "모호성 A와 추적 오차 e 사이의 함수 형태"를 데이터에서 적합.

왜 필요한가: P1~P3는 전부 **상관·순서 검정**이다("A가 크면 e도 크다", "A가 먼저 온다").
논문 §4가 "검증이지 구현이 아니다"라고 못박았지만, 마지막 절의 **설계 함의**(망각 있는 추정기 /
죽은 시간에 묶인 게인)를 쓰려면 "얼마나?"에 답하는 관계식이 있어야 한다. 그리고 그 형태가
제어이론 언어로 읽히면 사용자의 로보틱스 방향과도 이어진다(RESEARCH_DEFINITION.md §0).

여기서 적합하는 두 가지:

**(1) 정적 관계 e(A)** — 어떤 함수 형태가 데이터에 맞는가.
  - 선형:        e = a·A + b
  - 발산형:      e = a/(1-A) + b        ← A→1(완전히 구별 불가)에서 오차가 발산한다는 가설
  - 로그 발산형: e = -a·log(1-A) + b    ← 더 완만한 발산
  발산형이 선형보다 뚜렷이 잘 맞으면, "모호성은 선형 난이도 축이 아니라 **극한에서 터지는**
  축"이라는 정량적 진술이 되고, 이게 §2의 "관측 우도가 평평하면 사후분포가 예측모델에
  지배된다"는 메커니즘과 같은 이야기다.

**(2) 동적 관계 — 1차 제어 모델의 보정 게인 K(A)**
  추적기를 "오차를 관측으로 당기는 비례 보정기 + 템포 외삽 편향"으로 보면
      de/dt = -K·e + c
  이고, §2의 메커니즘 주장("모호하면 관측이 보정을 못 해서 개루프 적분기가 된다")은 곧
      **A가 커질수록 K → 0**
  이라는 말이다. 이건 상관계수가 아니라 **모수**라서 직접 적합해 확인할 수 있다.
  구현: 모호성 구간(bin)별로 (e, de/dt) 점들을 모아 de/dt = -K·e + c 를 최소제곱 적합.
  K가 A에 대해 단조감소하면 메커니즘 주장이 모수 수준에서 지지된다.
  부산물: 1/K = 그 구간에서 오차가 저절로 줄어드는 시간상수(초). K→0이면 시간상수가 무한 =
  "한 번 틀리면 스스로 못 돌아온다"는 개루프 상태.
"""
import numpy as np


def _r2(y, y_hat):
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_linear(x, y):
    """y = a*x + b. 반환: dict(a, b, r2, form)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(form="e = a*A + b", a=float(coef[0]), b=float(coef[1]),
                r2=_r2(y, A @ coef))


def fit_divergent(x, y, eps=1e-3):
    """y = a/(1-x) + b. x는 [0,1) 범위의 모호성(maxsim)이라고 가정."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = 1.0 / np.maximum(1.0 - x, eps)
    A = np.column_stack([z, np.ones_like(z)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(form="e = a/(1-A) + b", a=float(coef[0]), b=float(coef[1]),
                r2=_r2(y, A @ coef))


def fit_log_divergent(x, y, eps=1e-3):
    """y = -a*log(1-x) + b."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = -np.log(np.maximum(1.0 - x, eps))
    A = np.column_stack([z, np.ones_like(z)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(form="e = -a*log(1-A) + b", a=float(coef[0]), b=float(coef[1]),
                r2=_r2(y, A @ coef))


def fit_static_forms(amb, err):
    """세 가지 함수 형태를 모두 적합해서 r2로 비교. 반환: list[dict]"""
    return [fit_linear(amb, err), fit_divergent(amb, err), fit_log_divergent(amb, err)]


def binned_medians(amb, err, n_bins=10):
    """모호성을 분위수로 n_bins개 구간으로 나눠, 구간별 (모호성 중앙값, 오차 중앙값, n).

    프레임 단위 원자료는 분산이 커서 함수 형태 비교가 잡음에 묻힌다. 구간 중앙값으로
    요약하면 형태 비교가 안정된다(중앙값이라 이상치에도 강함).
    """
    amb, err = np.asarray(amb, float), np.asarray(err, float)
    qs = np.quantile(amb, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (amb >= qs[i]) & (amb < qs[i + 1])
        if m.sum() < 5:
            continue
        out.append((float(np.median(amb[m])), float(np.median(err[m])), int(m.sum())))
    return out


def fit_gain_by_ambiguity(amb, err, hop_sec, n_bins=5, smooth_frames=1):
    """동적 관계: 모호성 구간별로 de/dt = -K*e + c 를 적합해 보정 게인 K를 뽑는다.

    반환: list[dict(amb_median, K, c, tau_sec, r2, n)]
      - K: 보정 게인 (1/초). 클수록 오차를 관측으로 빨리 되돌린다.
      - tau_sec = 1/K: 시간상수. K<=0이면 inf(=스스로 회복 안 됨, 개루프)
    ⚠️ 시계열 하나가 아니라 여러 곡의 프레임을 모아 적합하는 것이므로, 각 곡의 궤적은
    미리 이어붙이지 말고 곡별로 미분을 구해 넘겨야 한다(_concat_derivatives 참조).
    """
    amb, err = np.asarray(amb, float), np.asarray(err, float)
    dedt = np.gradient(err) / hop_sec
    qs = np.quantile(amb, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (amb >= qs[i]) & (amb < qs[i + 1])
        if m.sum() < 20:
            continue
        e_, d_ = err[m], dedt[m]
        A = np.column_stack([e_, np.ones_like(e_)])
        coef, *_ = np.linalg.lstsq(A, d_, rcond=None)
        slope, c = float(coef[0]), float(coef[1])
        K = -slope
        out.append(dict(amb_median=float(np.median(amb[m])), K=K, c=c,
                        tau_sec=(1.0 / K if K > 0 else float("inf")),
                        r2=_r2(d_, A @ coef), n=int(m.sum())))
    return out


def fit_gain_per_piece(traces, hop_sec, smooth_frames=20, amb_split="median"):
    """곡마다 **모호성 낮은 절반 / 높은 절반**에서 각각 K를 적합해 짝지어 비교한다.

    왜 이렇게 바꿨나(2026-09-17, 1차 시도 실패 후): `fit_gain_from_traces`로 여러 곡을
    합쳐 모호성 구간별 K를 적합했더니 r2가 0.00~0.01로 **모델이 데이터를 전혀 설명하지
    못했다**. 원인 두 가지로 추정:
      (a) 오차 e가 프레임 인덱스 단위(0.1초 계단)로 양자화돼 있어, 매끄럽게 하지 않고
          미분하면 de/dt가 사실상 전부 양자화 잡음이 된다.
      (b) 여러 곡을 한 구간에 섞으면 곡마다 다른 동역학이 서로 상쇄된다.
    그래서 (a)는 미분 전 이동평균으로, (b)는 **곡 내 짝비교**로 고친다.

    반환: list[dict(label_idx, amb_lo, amb_hi, K_lo, K_hi, tau_lo, tau_hi, r2_lo, r2_hi, n)]
      - K_lo: 그 곡의 모호성 낮은 절반에서의 보정 게인
      - K_hi: 높은 절반에서의 보정 게인
      가설이 맞으면 곡마다 K_hi < K_lo (모호하면 보정이 죽는다).
    """
    out = []
    for idx, (amb, err) in enumerate(traces):
        amb, err = np.asarray(amb, float), np.asarray(err, float)
        if len(err) < max(100, 4 * smooth_frames):
            continue
        e_s = _moving_average(err, smooth_frames)
        dedt = np.gradient(e_s) / hop_sec
        thresh = np.median(amb) if amb_split == "median" else amb_split
        lo_m, hi_m = amb <= thresh, amb > thresh
        if lo_m.sum() < 30 or hi_m.sum() < 30:
            continue
        res = {}
        for tag, m in (("lo", lo_m), ("hi", hi_m)):
            e_, d_ = e_s[m], dedt[m]
            if np.std(e_) < 1e-9:
                res[f"K_{tag}"], res[f"r2_{tag}"] = np.nan, np.nan
                continue
            A = np.column_stack([e_, np.ones_like(e_)])
            coef, *_ = np.linalg.lstsq(A, d_, rcond=None)
            K = -float(coef[0])
            res[f"K_{tag}"] = K
            res[f"c_{tag}"] = float(coef[1])
            res[f"r2_{tag}"] = _r2(d_, A @ coef)
            res[f"tau_{tag}"] = 1.0 / K if K > 0 else float("inf")
        res.update(piece_idx=idx, amb_lo=float(np.median(amb[lo_m])),
                   amb_hi=float(np.median(amb[hi_m])), n=int(len(err)))
        out.append(res)
    return out


def _moving_average(x, window_frames):
    x = np.asarray(x, dtype=float)
    k = int(window_frames)
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    pad = k // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(padded, kernel, mode="valid")


def concat_piece_traces(traces, hop_sec):
    """곡별 (amb, err) 궤적 리스트를 받아 (amb_all, err_all, dedt_all)로 이어붙인다.

    미분은 **곡 경계를 넘지 않게** 곡마다 따로 계산한다 — 그냥 이어붙이고 미분하면
    곡이 바뀌는 지점에서 거짓 점프가 생긴다(흔한 실수라 여기서 막는다).
    """
    ambs, errs, dedts = [], [], []
    for amb, err in traces:
        amb, err = np.asarray(amb, float), np.asarray(err, float)
        if len(err) < 3:
            continue
        ambs.append(amb)
        errs.append(err)
        dedts.append(np.gradient(err) / hop_sec)
    if not ambs:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(ambs), np.concatenate(errs), np.concatenate(dedts)


def fit_gain_from_traces(traces, hop_sec, n_bins=5):
    """concat_piece_traces + 구간별 게인 적합. 곡 경계 안전 버전."""
    amb, err, dedt = concat_piece_traces(traces, hop_sec)
    if len(amb) == 0:
        return []
    qs = np.quantile(amb, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (amb >= qs[i]) & (amb < qs[i + 1])
        if m.sum() < 20:
            continue
        e_, d_ = err[m], dedt[m]
        A = np.column_stack([e_, np.ones_like(e_)])
        coef, *_ = np.linalg.lstsq(A, d_, rcond=None)
        K = -float(coef[0])
        out.append(dict(amb_median=float(np.median(amb[m])), K=K, c=float(coef[1]),
                        tau_sec=(1.0 / K if K > 0 else float("inf")),
                        r2=_r2(d_, A @ coef), n=int(m.sum())))
    return out
