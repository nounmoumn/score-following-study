"""Controlled loss of position evidence; synthetic mechanism, not natural P3."""
import numpy as np
from .online_dtw import OnlineDTW


class CostMatrixDTW(OnlineDTW):
    """Same recurrence and decisions, with precomputed original cosine costs.

    Erasure replaces each selected cost row by its original mean, preserving
    average mismatch while removing its dependence on score position.
    """
    def __init__(self, u, v, *, erase=None, **kwargs):
        super().__init__(u, v, **kwargs)
        un = np.linalg.norm(u, axis=1)
        vn = np.linalg.norm(v, axis=1)
        norm = un[:, None] * vn[None, :]
        self.costs = np.full((len(u), len(v)), 2.0)
        np.divide(u @ v.T, norm, out=self.costs, where=norm > 0)
        self.costs = np.where(norm > 0, 1.0 - self.costs, 2.0)
        if erase is not None:
            lo, hi = erase
            self.costs[lo:hi] = self.costs[lo:hi].mean(axis=1, keepdims=True)

    def _cost(self, i, j):
        return float(self.costs[i, j])


def synthetic_pair(seed, speed_change, *, frames=450, onset=150):
    """Random held notes; tempo perturbation starts at preassigned onset."""
    rng = np.random.default_rng(seed)
    n_score = int(np.ceil(frames * 1.3)) + 10
    pitches = []
    while len(pitches) < n_score:
        pitches.extend([int(rng.integers(12))] * int(rng.integers(2, 7)))
    v = np.eye(12)[np.asarray(pitches[:n_score])]
    t = np.arange(frames)
    truth = t + speed_change * np.maximum(0, t - onset)
    u = v[np.floor(truth).astype(int)]
    return u, v, truth


def replay_error(u, v, truth, *, hop, band, erase=None):
    tracker = CostMatrixDTW(u, v, band_width=band, erase=erase)
    tracker.run(stop_at_performance_end=True)
    estimate = np.full(len(u), np.nan)
    estimate[0] = 0
    for t, j in tracker.real_path():
        estimate[t] = j
    if not np.isfinite(estimate).all():
        raise RuntimeError("tracker omitted a performance frame")
    return abs(estimate - truth) * hop
