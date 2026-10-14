import numpy as np

from score_following.relations import (
    binned_medians,
    concat_piece_traces,
    fit_divergent,
    fit_gain_from_traces,
    fit_linear,
    fit_static_forms,
)


def test_fit_linear_recovers_known_coefficients():
    x = np.linspace(0, 0.9, 200)
    y = 3.0 * x + 1.5
    f = fit_linear(x, y)
    assert np.isclose(f["a"], 3.0)
    assert np.isclose(f["b"], 1.5)
    assert f["r2"] > 0.999


def test_fit_divergent_wins_on_divergent_data():
    """데이터가 실제로 1/(1-A) 형태면 발산형 적합이 선형보다 r2가 높아야 한다.
    (이게 뒤집히면 함수 형태 비교 자체가 무의미해진다)"""
    rng = np.random.default_rng(0)
    A = np.linspace(0.0, 0.95, 400)
    y = 0.2 / (1 - A) + 0.5 + rng.normal(scale=0.05, size=A.size)
    forms = {f["form"]: f for f in fit_static_forms(A, y)}
    r2_lin = forms["e = a*A + b"]["r2"]
    r2_div = forms["e = a/(1-A) + b"]["r2"]
    assert r2_div > r2_lin + 0.05, f"발산형 r2={r2_div:.3f}, 선형 r2={r2_lin:.3f}"


def test_fit_linear_wins_on_linear_data():
    """반대 방향도 확인 — 진짜 선형 데이터면 선형이 이겨야 한다(적합이 발산형에 편향되지 않음)."""
    rng = np.random.default_rng(1)
    A = np.linspace(0.0, 0.95, 400)
    y = 2.0 * A + 0.3 + rng.normal(scale=0.05, size=A.size)
    forms = {f["form"]: f for f in fit_static_forms(A, y)}
    assert forms["e = a*A + b"]["r2"] > forms["e = a/(1-A) + b"]["r2"]


def test_binned_medians_monotone_input_gives_monotone_output():
    A = np.linspace(0, 1, 500)
    e = A * 4
    bins = binned_medians(A, e, n_bins=5)
    ambs = [b[0] for b in bins]
    errs = [b[1] for b in bins]
    assert ambs == sorted(ambs)
    assert errs == sorted(errs)


def test_concat_traces_does_not_differentiate_across_piece_boundary():
    """곡 경계에서 거짓 점프가 생기지 않아야 한다 — 두 곡을 이어붙였을 때
    미분 최대값이 곡 내부 기울기 수준이어야 한다(경계 점프면 훨씬 커진다)."""
    hop = 0.1
    piece_a = (np.full(50, 0.5), np.linspace(0, 1, 50))     # 기울기 = 1/(50*0.1)=0.2
    piece_b = (np.full(50, 0.9), np.linspace(100, 101, 50))  # 값 자체가 100 이상 큼
    _, _, dedt = concat_piece_traces([piece_a, piece_b], hop)
    assert np.max(np.abs(dedt)) < 1.0, f"경계 점프 의심, max|de/dt|={np.max(np.abs(dedt))}"


def test_fit_gain_recovers_known_first_order_dynamics():
    """de/dt = -K*e + c 로 생성한 궤적에서 K를 되찾는지.
    낮은 모호성 구간은 K가 크고(빨리 회복), 높은 구간은 K≈0(개루프)이 되도록 만들었다."""
    hop = 0.1
    rng = np.random.default_rng(2)
    traces = []
    for amb_level, K_true, c in [(0.3, 2.0, 0.05), (0.9, 0.0, 0.3)]:
        e = [0.1]
        for _ in range(600):
            de = -K_true * e[-1] + c
            e.append(max(0.0, e[-1] + de * hop + rng.normal(scale=0.002)))
        traces.append((np.full(len(e), amb_level), np.array(e)))
    fits = fit_gain_from_traces(traces, hop, n_bins=2)
    fits = sorted(fits, key=lambda d: d["amb_median"])
    K_low_amb, K_high_amb = fits[0]["K"], fits[1]["K"]
    assert K_low_amb > 1.0, f"낮은 모호성 구간 K={K_low_amb:.3f}, 기대 ≈2.0"
    assert K_high_amb < 0.5, f"높은 모호성 구간 K={K_high_amb:.3f}, 기대 ≈0"
    assert fits[0]["tau_sec"] < fits[1]["tau_sec"]
