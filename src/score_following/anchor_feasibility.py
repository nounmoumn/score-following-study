"""Offline perfect-anchor diagnostic; selectors never receive ground truth."""
import numpy as np
from scipy.stats import rankdata

from .observation_evidence import predict_prior
from .fixed_lag_smoothing import backward_adjoint


def smooth(cost, anchors=None, transition=(.2, .65, .1, .05), beta=8., restart=1e-6):
    likelihood = np.exp(-beta*(cost-cost.min(axis=1, keepdims=True)))
    m, n = likelihood.shape
    for frame, state in (anchors or {}).items():
        likelihood[frame] = 0.
        likelihood[frame, state] = 1.
    forward = np.empty_like(likelihood)
    p = np.zeros(n)
    p[0] = 1.
    for t in range(m):
        q = predict_prior(p, transition, restart) if t else p
        p = q*likelihood[t]
        if p.sum() <= 0:
            raise ValueError("impossible anchor")
        p = p / p.sum()
        forward[t] = p
    b = np.ones(n)
    for t in range(m-1, -1, -1):
        p = forward[t]*b
        forward[t] = p/p.sum()
        if t:
            b = backward_adjoint(likelihood[t]*b, transition, restart)
            b /= b.max()
    return forward


def candidate_features(posterior, ambiguity, indices):
    rows = posterior[indices]
    center = rows.argmax(axis=1)
    confidence = np.array([r[max(0, c-5):c+6].sum() for r,c in zip(rows,center)])
    valid = np.isfinite(ambiguity)
    mass = rows[:, valid].sum(1)
    a = rows[:,valid] @ ambiguity[valid] / np.maximum(mass,1e-300)
    arank = rankdata(a, method="average") / len(a)
    arank[mass < .99] = .5
    urank = rankdata(1-confidence, method="average") / len(a)
    return arank, urank


def choose(times, a, u, budget, policy, seed=0):
    if policy == "uniform":
        targets = np.linspace(times[0],times[-1],budget+2)[1:-1]
        return np.array([int(np.argmin(abs(times-t))) for t in targets])
    rng = np.random.default_rng(seed)
    score = {"ambiguity": a, "uncertainty": u, "joint": (a+u)/2,
             "sham_joint": (rng.permutation(a)+u)/2,
             "random": rng.random(len(times))}[policy]
    selected = []
    for i in np.argsort(-score, kind="stable"):
        if all(abs(times[i]-times[j]) >= 5 for j in selected):
            selected.append(int(i))
        if len(selected) == budget:
            return np.array(selected)
    raise ValueError("cannot meet anchor budget")
