"""Fixed-lag smoothing on the audit's own filter, for the P2 information contrast.

Only the amount of future observation changes: same states, same fixed-step
transition with the same restart mass, same cosine cost. Lag 0 is the filter by
construction, which is the equivalence check the protocol requires.

Smoothing reads future performance frames. Nothing here is a live result.
"""
import numpy as np

from .observation_evidence import predict_prior


def backward_adjoint(g, transition, restart):
    """Adjoint of `predict_prior`: returns b(i) = sum_j A(i,j) g(j).

    `predict_prior` sends state i to i+step, piling every i >= n-step onto n-1, then
    mixes in uniform restart mass. This is the exact transpose of that, so applied to
    an all-ones vector it returns all ones.
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    out = np.zeros(n)
    for step, weight in enumerate(transition):
        if step == 0:
            out += weight*g
        else:
            out[:n-step] += weight*g[step:]
            out[n-step:] += weight*g[n-1]
    return (1-restart)*out + restart*g.sum()/n


def smoothed_trace(cost, lag, *, score_hop=.1, tolerance=.5,
                   transition=(.2, .65, .1, .05), beta=8., restart=1e-6):
    """Estimate for frame t-lag using observations up to frame t.

    Returns arrays indexed by the ESTIMATED frame, length len(cost)-lag, plus the
    emitting frame index so the caller can charge the delay honestly. No annotation,
    truth or performance duration is an argument.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("invalid observation matrix")
    if not isinstance(lag, (int, np.integer)) or lag < 0 or lag >= len(cost):
        raise ValueError("invalid lag")
    if not np.isclose(sum(transition), 1) or min(transition) < 0 or beta <= 0:
        raise ValueError("invalid filter settings")
    m, n = cost.shape
    likelihood = np.exp(-beta*(cost - cost.min(axis=1, keepdims=True)))
    # Forward filter, identical to confidence_trace's recursion.
    forward = np.empty((m, n))
    posterior = np.zeros(n)
    posterior[0] = 1.
    for frame in range(m):
        prior = predict_prior(posterior, transition, restart) if frame else posterior.copy()
        posterior = prior*likelihood[frame]
        posterior /= posterior.sum()
        forward[frame] = posterior
    count = m - lag
    states = np.arange(n)
    fields = ("map_index", "position_score_sec", "confidence", "pmax",
              "emitting_frame", "estimated_frame")
    result = {key: np.empty(count) for key in fields}
    for target in range(count):
        emitting = target + lag
        b = np.ones(n)
        # Backward over the window only: never reads a frame beyond `emitting`.
        for s in range(emitting, target, -1):
            b = backward_adjoint(likelihood[s]*b, transition, restart)
            b /= b.max()
        smoothed = forward[target]*b
        total = smoothed.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid smoothing mass")
        smoothed /= total
        center = int(smoothed.argmax())
        region = np.abs(states-center)*score_hop <= tolerance + 1e-12
        values = (center, (center+1)*score_hop, float(np.clip(smoothed[region].sum(), 0, 1)),
                  float(smoothed[center]), emitting, target)
        for key, value in zip(fields, values):
            result[key][target] = value
    return result
