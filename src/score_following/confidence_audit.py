"""GT-free confidence instrumentation and separate retrospective evaluation."""
from collections import Counter, defaultdict

import numpy as np

from .observation_evidence import predict_prior


def blocked_maxsim(chroma, window, exclude, block=256):
    """Identical defined values/indexing to ambiguity.py, without dense N^2 storage."""
    n = len(chroma)
    count = n - window
    out = np.full(n, np.nan)
    if count <= 0:
        return out, 0
    features = np.stack([chroma[i:i + window].reshape(-1) for i in range(count)])
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    silent = int(np.sum(norm[:, 0] == 0))
    features = features / np.where(norm > 0, norm, 1.)
    for start in range(0, count, block):
        end = min(count, start + block)
        sim = features[start:end] @ features.T
        for local, i in enumerate(range(start, end)):
            sim[local, max(0, i-exclude):min(count, i+exclude+1)] = -np.inf
        maxima = sim.max(axis=1)
        out[start:end] = np.where(np.isfinite(maxima), maxima, np.nan)
    return out, silent


def logit(value, clip=1e-12):
    """Kept for the v1 definition and its tests. Saturates for value near 1."""
    value = np.clip(value, clip, 1.-clip)
    return np.log(value) - np.log1p(-value)


def regional_log_odds(prior, unnormalized, region, clip=1e-12):
    """Exact log odds of a state set, by direct tail summation.

    v1 computed logit(mass) from the normalized mass, so 1-mass was lost to
    cancellation once the filter concentrated: 13.0% of real observations had
    both masses clipped and returned evidence exactly 0. Summing the complement
    directly avoids that, and dividing unnormalized posterior sums cancels the
    normalizer, so the posterior odds need no separate normalization.

    An empty complement, or a prior with no mass outside the set (the one-hot
    initial frame), is a STRUCTURAL degeneracy, not a numerical one: the odds
    are infinite and the observation cannot move a set that already holds every
    unit of mass. Those frames are flagged so later work can drop them instead
    of reading a fabricated zero.
    """
    bound = float(np.log1p(-clip) - np.log(clip))
    p_in, p_out = float(unnormalized[region].sum()), float(unnormalized[~region].sum())
    u_in, u_out = float(prior[region].sum()), float(prior[~region].sum())
    if p_out <= 0 or u_out <= 0 or p_in <= 0 or u_in <= 0:
        return bound, bound, 1
    return np.log(p_in)-np.log(p_out), np.log(u_in)-np.log(u_out), 0


def confidence_trace(cost, ambiguity, *, score_hop=.1, tolerance=.5,
                     transition=(.2, .65, .1, .05), beta=8., restart=1e-6,
                     clip=1e-12):
    """No truth, annotations, performance duration or outcomes are arguments."""
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(cost).all() or cost.shape[1] < len(transition):
        raise ValueError("invalid observation matrix")
    if len(ambiguity) != cost.shape[1] or score_hop <= 0 or tolerance < 0:
        raise ValueError("invalid score grid")
    if not np.isclose(sum(transition), 1) or min(transition) < 0 or beta <= 0 or not 0 <= restart < 1:
        raise ValueError("invalid filter settings")
    n = cost.shape[1]
    posterior = np.zeros(n)
    posterior[0] = 1.
    valid = np.isfinite(ambiguity)
    fields = ("map_index", "position_score_sec", "confidence", "prior_confidence",
              "evidence", "confidence_log_odds", "prior_log_odds", "evidence_saturated",
              "flat_likelihood", "pmax", "entropy", "state_margin", "ambiguity",
              "ambiguity_valid_mass")
    result = {key: np.empty(len(cost)) for key in fields}
    states = np.arange(n)
    for frame, row in enumerate(cost):
        prior = predict_prior(posterior, transition, restart) if frame else posterior.copy()
        likelihood = np.exp(-beta*(row-row.min()))
        unnormalized = prior * likelihood
        posterior = unnormalized/unnormalized.sum()
        center = int(posterior.argmax())
        region = np.abs(states-center)*score_hop <= tolerance + 1e-12
        c, u = float(posterior[region].sum()), float(prior[region].sum())
        mass = float(posterior[valid].sum())
        nonzero = posterior[posterior > 0]
        top = np.partition(posterior, -2)[-2:]
        odds_c, odds_u, saturated = regional_log_odds(prior, unnormalized, region, clip)
        # A constant cost row means the observation cannot discriminate any state.
        # cosine_cost produces one whenever the performance frame has zero energy,
        # so this is an ordinary runtime-observable condition, not an error path.
        flat = bool(np.all(row == row[0]))
        evidence = 0. if flat or saturated else float(odds_c-odds_u)
        values = (center, (center+1)*score_hop, np.clip(c, 0, 1), np.clip(u, 0, 1),
                  evidence, odds_c, odds_u, saturated, int(flat), float(posterior[center]),
                  float(-(nonzero*np.log(nonzero)).sum()/np.log(n)),
                  float(top.max()-top.min()),
                  float(posterior[valid] @ ambiguity[valid] / mass) if mass > 0 else 0.,
                  mass)
        for key, value in zip(fields, values):
            result[key][frame] = value
    return result


def anchor_labels(trace, perf_beats, score_beats, *, hop=.1, tolerance=.5):
    """GT is introduced ONLY after inference. Select outputs already available."""
    perf_beats, score_beats = np.asarray(perf_beats), np.asarray(score_beats)
    if (len(perf_beats) != len(score_beats) or len(perf_beats) < 2
            or not np.isfinite(perf_beats).all() or not np.isfinite(score_beats).all()
            or np.any(np.diff(perf_beats) <= 0) or np.any(np.diff(score_beats) <= 0)):
        raise ValueError("invalid/unrepaired annotation clock")
    available = (np.arange(len(trace["map_index"]))+1)*hop
    indices = np.searchsorted(available, perf_beats, side="right")-1
    mask = (indices >= 0) & (perf_beats <= available[-1])
    idx = indices[mask]
    truth = score_beats[mask]
    error = np.abs(trace["position_score_sec"][idx]-truth)
    return {"indices": idx, "times": perf_beats[mask], "truth": truth,
            "ages": perf_beats[mask]-available[idx], "error": error,
            "correct": (error <= tolerance+1e-12).astype(int)}


def work_weights(rows):
    """Equal works, equal performances within work, equal samples within performance."""
    counts = Counter((r["work_family"], r["case_id"]) for r in rows)
    families = defaultdict(set)
    for family, case in counts:
        families[family].add(case)
    return np.array([1./len(families)/len(families[r["work_family"]])/
                     counts[(r["work_family"], r["case_id"])] for r in rows])


def summarize(rows, weights=None, bins=10, high=.9, clip=1e-12):
    if not rows:
        return {"observations": 0}
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    w = w/w.sum()
    c = np.array([r["confidence"] for r in rows], dtype=float)
    y = np.array([r["correct"] for r in rows], dtype=float)
    if not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError("invalid confidence")
    c_log = np.clip(c, clip, 1-clip)
    ids = np.minimum((c*bins).astype(int), bins-1)
    reliability, ece = [], 0.
    for i in range(bins):
        mask = ids == i
        mass = float(w[mask].sum())
        if not mask.any():
            continue
        pred, acc = float(w[mask] @ c[mask]/mass), float(w[mask] @ y[mask]/mass)
        ece += mass*abs(pred-acc)
        reliability.append({"bin": i, "lower": i/bins, "upper": (i+1)/bins,
                            "observations": int(mask.sum()), "weight": mass,
                            "confidence": pred, "accuracy": acc})
    selected = c >= high
    coverage = float(w[selected].sum())
    return {"observations": len(rows), "accuracy": float(w@y),
            "mean_confidence": float(w@c), "brier": float(w@((c-y)**2)),
            "logloss": float(-w@(y*np.log(c_log)+(1-y)*np.log1p(-c_log))),
            "ece": float(ece), "high_confidence_coverage": coverage,
            "high_confidence_accuracy": float(w[selected]@y[selected]/coverage) if coverage else None,
            "reliability": reliability}
