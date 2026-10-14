import numpy as np
import pytest

from score_following.confidence_audit import confidence_trace
from score_following.fixed_lag_smoothing import backward_adjoint, smoothed_trace
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import bayes_from_cost

TRANSITION = (.2, .65, .1, .05)


def dense_kernel(n, transition, restart):
    """A(i,j) built by pushing each basis vector through the forward operator."""
    rows = []
    for i in range(n):
        p = np.zeros(n)
        p[i] = 1.
        rows.append(predict_prior(p, transition, restart))
    return np.array(rows)


def test_backward_operator_is_the_exact_transpose_of_the_forward_one():
    n, restart = 12, 1e-3
    a = dense_kernel(n, TRANSITION, restart)
    rng = np.random.default_rng(5)
    g = rng.random(n)
    np.testing.assert_allclose(backward_adjoint(g, TRANSITION, restart), a @ g, atol=1e-12)


def test_backward_operator_preserves_the_all_ones_vector():
    for n in (5, 12, 400):
        for restart in (0., 1e-6, 1e-2):
            out = backward_adjoint(np.ones(n), TRANSITION, restart)
            np.testing.assert_allclose(out, 1., atol=1e-12)


def test_zero_lag_reproduces_the_audit_filter_exactly():
    rng = np.random.default_rng(17)
    cost = rng.random((50, 40))
    smooth = smoothed_trace(cost, 0, transition=TRANSITION)
    filtered = confidence_trace(cost, np.zeros(40), transition=TRANSITION)
    np.testing.assert_array_equal(smooth["map_index"], filtered["map_index"])
    np.testing.assert_array_equal(smooth["map_index"], bayes_from_cost(cost, TRANSITION))
    np.testing.assert_allclose(smooth["confidence"], filtered["confidence"], atol=1e-12)
    np.testing.assert_allclose(smooth["position_score_sec"], filtered["position_score_sec"])
    np.testing.assert_array_equal(smooth["emitting_frame"], smooth["estimated_frame"])


def test_smoothing_never_reads_beyond_its_emitting_frame():
    rng = np.random.default_rng(23)
    cost = rng.random((60, 30))
    lag = 7
    full = smoothed_trace(cost, lag, transition=TRANSITION)
    # Truncating the input right after an estimate's emitting frame must not move it.
    cut = 40
    truncated = smoothed_trace(cost[:cut+1], lag, transition=TRANSITION)
    keep = cut+1-lag
    for field in ("map_index", "confidence", "position_score_sec"):
        np.testing.assert_allclose(full[field][:keep], truncated[field], atol=1e-10)


def test_emission_delay_is_exactly_the_lag():
    cost = np.random.default_rng(1).random((30, 20))
    for lag in (0, 1, 5, 9):
        out = smoothed_trace(cost, lag, transition=TRANSITION)
        assert len(out["map_index"]) == 30-lag
        np.testing.assert_array_equal(out["emitting_frame"]-out["estimated_frame"], lag)


def brute_force_fixed_lag(cost, lag, transition, restart, beta):
    """p(x_{t-lag} | y_1..y_t) by explicit dense matrices. Reference implementation."""
    m, n = cost.shape
    a = dense_kernel(n, transition, restart)
    likelihood = np.exp(-beta*(cost - cost.min(axis=1, keepdims=True)))
    forward = np.empty((m, n))
    p = np.zeros(n)
    p[0] = 1.
    for t in range(m):
        prior = a.T @ forward[t-1] if t else p
        q = prior*likelihood[t]
        forward[t] = q/q.sum()
    out = []
    for target in range(m-lag):
        emitting = target+lag
        b = np.ones(n)
        for s in range(emitting, target, -1):
            b = a @ (likelihood[s]*b)
        joint = forward[target]*b
        out.append(joint/joint.sum())
    return np.array(out)


@pytest.mark.parametrize("lag", [0, 1, 3, 6])
def test_matches_a_brute_force_dense_forward_backward(lag):
    rng = np.random.default_rng(31)
    cost = rng.random((14, 18))
    reference = brute_force_fixed_lag(cost, lag, TRANSITION, 1e-3, 8.)
    out = smoothed_trace(cost, lag, transition=TRANSITION, restart=1e-3)
    np.testing.assert_array_equal(out["map_index"], reference.argmax(axis=1))
    np.testing.assert_allclose(out["pmax"], reference.max(axis=1), atol=1e-10)


def test_smoothing_conditions_on_more_data_so_it_concentrates_more():
    rng = np.random.default_rng(29)
    cost = rng.random((80, 50))
    filtered = smoothed_trace(cost, 0, transition=TRANSITION)
    for lag in (1, 5, 20):
        smooth = smoothed_trace(cost, lag, transition=TRANSITION)
        shared = len(smooth["pmax"])
        assert smooth["pmax"].mean() >= filtered["pmax"][:shared].mean()
        assert not np.array_equal(smooth["map_index"], filtered["map_index"][:shared])


def test_invalid_lag_and_matrix_are_refused():
    cost = np.ones((5, 10))
    with pytest.raises(ValueError):
        smoothed_trace(cost, 5)
    with pytest.raises(ValueError):
        smoothed_trace(cost, -1)
    with pytest.raises(ValueError):
        smoothed_trace(np.full((5, 10), np.nan), 1)
