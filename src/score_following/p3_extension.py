"""Frozen P3 extension: causal trackers, direct evidence, and paired cost interventions."""
import hashlib

import numpy as np
from numba import njit


@njit(cache=True)
def _evaluate(cost, d, length, i, j):
    if length[i, j] != 0:
        return
    local = cost[i, j]
    if i == 0 and j == 0:
        d[i, j], length[i, j] = local, 1
        return
    best, size = np.inf, 0
    if j > 0 and length[i, j-1] > 0:
        best, size = d[i, j-1] + local, length[i, j-1] + 1
    if i > 0 and length[i-1, j] > 0:
        value = d[i-1, j] + local
        if value < best:
            best, size = value, length[i-1, j] + 1
    if i > 0 and j > 0 and length[i-1, j-1] > 0:
        value = d[i-1, j-1] + 2*local
        if value < best:
            best, size = value, length[i-1, j-1] + 2
    if size:
        d[i, j], length[i, j] = best, size


@njit(cache=True)
def oltw_from_cost(cost, band, max_run=3, audit_candidates=None, audit_values=None):
    """Exact original OnlineDTW decisions, stopped at first final live frame.

    Numba is only an execution acceleration, not a new tracking algorithm.
    Full DP arrays are per-replay and released before the next condition.
    """
    m, n = cost.shape
    d = np.full((m, n), np.inf)
    length = np.zeros((m, n), np.int32)
    estimate = np.zeros(m, np.int64)
    t, j, previous, count = 0, 0, -1, 0
    _evaluate(cost, d, length, 0, 0)
    while t < m-1:
        if audit_candidates is not None and audit_values is not None:
            # Overwrite while the same live frame advances in score; final
            # snapshot at a frame is its last state before reading the next.
            # Read only cells already computed at this instant.
            for which in range(2):
                candidate = audit_candidates[t, which]
                audit_values[t, which] = np.nan
                if candidate >= max(0, j-band+1) and candidate <= j and length[t, candidate]>0:
                    audit_values[t, which] = d[t, candidate]/length[t, candidate]
            audit_values[t, 2] = j
        # Both=0, Row=1, Column=2, matching original tie order.
        inc = 0
        if t >= band:
            if count > max_run:
                inc = 2 if previous == 1 else 1
            else:
                best, x, y = np.inf, t, j
                for k in range(max(0, j-band), j+1):
                    if length[t, k] and d[t, k]/length[t, k] < best:
                        best, x, y = d[t, k]/length[t, k], t, k
                for k in range(max(0, t-band), t+1):
                    if length[k, j] and d[k, j]/length[k, j] < best:
                        best, x, y = d[k, j]/length[k, j], k, j
                if x == t and y < j:
                    inc = 1
                elif y == j and x < t:
                    inc = 2
        if inc != 2 and t+1 < m:
            t += 1
            for k in range(max(0, j-band+1), j+1):
                _evaluate(cost, d, length, t, k)
        if inc != 1 and j+1 < n:
            j += 1
            for k in range(max(0, t-band+1), t+1):
                _evaluate(cost, d, length, k, j)
        count = count+1 if inc == previous else 1
        if inc != 0:
            previous = inc
        estimate[t] = j
    return estimate


def bayes_from_cost(cost, transition=(.2, .65, .1, .05), beta=8., restart=1e-6):
    """Causal filtering MAP with fixed tempo transitions and small restart mass."""
    if not np.isclose(sum(transition), 1) or min(transition) < 0:
        raise ValueError("invalid transition probabilities")
    if not 0 <= restart < 1 or beta <= 0:
        raise ValueError("invalid filter settings")
    m, n = cost.shape
    posterior = np.zeros(n)
    posterior[0] = 1
    estimate = np.zeros(m, dtype=int)
    for i in range(m):
        if i:
            prior = np.zeros(n)
            for step, weight in enumerate(transition):
                if step == 0:
                    prior += weight*posterior
                else:
                    prior[step:] += weight*posterior[:-step]
                    prior[-1] += weight*posterior[-step:].sum()
            posterior = (1-restart)*prior + restart/n
        emission = np.exp(-beta*(cost[i] - cost[i].min()))
        posterior *= emission
        total = posterior.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid filtering mass")
        posterior /= total
        estimate[i] = np.argmax(posterior)
    return estimate


def cosine_cost(u, v):
    den = np.linalg.norm(u, axis=1)[:, None]*np.linalg.norm(v, axis=1)[None, :]
    sim = np.divide(u @ v.T, den, out=np.zeros_like(den), where=den > 0)
    return np.where(den > 0, 1-sim, 2.0)


def evidence_at(u, v, truth_frames, frame, *, window, exclude, radius, frontier, band):
    """Past-input evidence with oracle past GT warp; no future performance frames."""
    first = frame-window+1
    if first < 0:
        raise ValueError("insufficient history")
    pos = np.rint(truth_frames[first:frame+1]).astype(int)
    center = pos[-1]
    offsets = pos-center
    ends = np.arange(max(0, -offsets.min()), len(v)-max(0, offsets.max()))
    templates = v[ends[:, None] + offsets[None, :]].reshape(len(ends), -1)
    query = u[first:frame+1].reshape(-1)
    norm = np.linalg.norm(templates, axis=1)*np.linalg.norm(query)
    sim = np.divide(templates @ query, norm, out=np.zeros(len(ends)), where=norm>0)
    where = np.flatnonzero(ends == center)
    if not len(where):
        raise ValueError("GT reference outside score")
    correct = sim[where[0]]
    wrong = abs(ends-center) > exclude
    masks = dict(global_=wrong, local=wrong & (abs(ends-center)<=radius),
                 accessible=wrong & (ends>=max(0, frontier-band+1)) & (ends<=frontier))
    result = dict(correct_similarity=float(correct))
    for name, mask in masks.items():
        key = "global" if name == "global_" else name
        result[f"weakness_{key}"] = float(sim[mask].max()-correct) if mask.any() else np.nan
        result[f"candidates_{key}"] = int(mask.sum())
    return result


def score_window_features(v, window):
    n = len(v)-window  # same convention as original ambiguity.py
    if n < 1:
        raise ValueError("score too short")
    x = np.lib.stride_tricks.sliding_window_view(v, window, axis=0)[:n].transpose(0, 2, 1)
    x = x.reshape(n, -1).copy()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm>0)


def score_ambiguity_at(features, completed_end, window, exclude):
    start = int(np.floor(completed_end))-window
    if not 0 <= start < len(features):
        return np.nan
    similarity = features @ features[start]
    keep = abs(np.arange(len(features))-start) > exclude
    return float(similarity[keep].max()) if keep.any() else np.nan


def distractor_sign(performance):
    return 1 if hashlib.sha256(performance.encode()).digest()[0] % 2 else -1


def distractor_costs(base, truth, *, onset, end, offset, guard, radius, seed):
    """Paired targeted/scattered interventions; exactly matched L1 and row means."""
    target, scattered = base.copy(), base.copy()
    rng = np.random.default_rng(seed)
    changed, mass = 0, 0.0
    for i in range(onset, end):
        true_j = int(round(truth[i]))
        dest = np.arange(true_j+offset-guard, true_j+offset+guard+1)
        src = dest-offset
        good = (dest>=0)&(dest<base.shape[1])&(src>=0)&(src<base.shape[1])
        dest, src = dest[good], src[good]
        if np.any(abs(dest-true_j)<=guard):
            raise ValueError("distractor overlaps protected true region")
        delta = np.maximum(base[i, dest]-base[i, src], 0)
        target[i, dest] -= delta
        mass += float(delta.sum())
        changed += int(np.count_nonzero(delta))
        pool = np.arange(max(0, true_j-radius), min(base.shape[1], true_j+radius+1))
        pool = pool[abs(pool-true_j)>guard]
        available = np.ones(len(pool), dtype=bool)
        for reduction in np.sort(delta[delta>0])[::-1]:
            eligible = np.flatnonzero(available & (base[i, pool]>=reduction-1e-12))
            if not len(eligible):
                raise ValueError("cannot match intervention magnitude")
            choice = int(rng.choice(eligible))
            scattered[i, pool[choice]] -= reduction
            available[choice] = False
    # Validate numerical invariants instead of merely asserting conceptual ones.
    if not np.allclose(target.sum(axis=1), scattered.sum(axis=1), atol=1e-9, rtol=1e-12):
        raise AssertionError("unmatched row means")
    if min(target.min(), scattered.min()) < -1e-10:
        raise AssertionError("negative cost")
    for i in range(onset, end):
        j = int(round(truth[i]))
        sl = slice(max(0, j-guard), min(base.shape[1], j+guard+1))
        if not np.array_equal(target[i, sl], base[i, sl]) or not np.array_equal(scattered[i, sl], base[i, sl]):
            raise AssertionError("true match changed")
    return target, scattered, dict(changed_cells=changed, total_reduction=mass,
        max_row_sum_mismatch=float(np.max(abs(target.sum(axis=1)-scattered.sum(axis=1)))))


def cue_costs(target, truth, *, first, end, guard, strength, sham=False):
    cost = target.copy()
    for i in range(first, end):
        j = int(round(truth[i]))
        lo, hi = max(0, j-guard), min(cost.shape[1], j+guard+1)
        if sham:
            cost[i] += strength*(1-(hi-lo)/cost.shape[1])
        else:
            cost[i] += strength
            cost[i, lo:hi] -= strength
    return cost


def sign_flip_p(effects, seed, n=19999):
    x = np.asarray(effects, dtype=float)
    if len(x)<2 or not np.isfinite(x).all():
        raise ValueError("invalid independent effects")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1., 1.]), size=(n, len(x)))
    return float((1+np.sum((signs*x).mean(axis=1)>=x.mean()-1e-14))/(n+1))


def holm_adjust(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    sorted_adjusted = np.minimum(1., np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result = np.empty_like(sorted_adjusted)
    result[order] = sorted_adjusted
    return result
