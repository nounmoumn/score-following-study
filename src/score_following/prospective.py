"""P3 prospective diagnostics; deliberately separate from historical P3 scoring."""
import numpy as np


def completed_window_values(score_time, window_end, ambiguity):
    """Latest saved score window wholly completed by each query time.

    No extrapolation before the first window, no interpolation from a future
    window. GT supplies score_time: this is an oracle analysis, not deployment.
    """
    score_time = np.asarray(score_time)
    end = np.asarray(window_end)
    if np.any(np.diff(end) < 0):
        raise ValueError("window ends must be monotone")
    idx = np.searchsorted(end, score_time + 1e-10, side="right") - 1
    out = np.full(score_time.shape, np.nan)
    valid = idx >= 0
    out[valid] = np.asarray(ambiguity)[idx[valid]]
    return out


def ridge_predict(x, y, test, weights, alpha=1.0):
    """Train-only weighted standardization and fixed ridge; intercept unpenalized."""
    weights = np.asarray(weights, dtype=float)
    weights = weights * len(weights) / weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-10] = 1.0
    xx = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    tt = np.column_stack([np.ones(len(test)), (test - mean) / scale])
    penalty = np.eye(xx.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(xx.T @ (weights[:, None] * xx) + penalty,
                           xx.T @ (weights * y))
    return tt @ coef


def grouped_predictions(x, y, groups, alpha=1.0):
    """Leave one entire composition out, including all its performances."""
    groups = np.asarray(groups)
    names, counts = np.unique(groups, return_counts=True)
    if len(names) < 3:
        raise ValueError("at least three compositions required")
    weight_map = dict(zip(names, 1.0 / counts))
    weights = np.array([weight_map[g] for g in groups])
    prediction = np.full(len(y), np.nan)
    for group in names:
        test = groups == group
        prediction[test] = ridge_predict(x[~test], y[~test], x[test],
                                          weights[~test], alpha)
    return prediction


def group_means(values, groups):
    return np.array([np.mean(values[groups == g]) for g in np.unique(groups)])


def bootstrap_mean(values, seed=42, n=10000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite independent units")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def shift_within_performance(features, performances, rng):
    """Jointly move ambiguity columns, never across performance boundaries."""
    shifted = features.copy()
    for perf in np.unique(performances):
        idx = np.flatnonzero(performances == perf)
        if len(idx) < 4:
            raise ValueError("too few landmarks for circular-shift diagnostic")
        lo = max(1, int(np.ceil(len(idx) * 0.2)))
        offset = int(rng.integers(lo, len(idx) - lo + 1))
        shifted[idx] = np.roll(features[idx], offset, axis=0)
    return shifted


def forecast_comparison(base, extra, y, groups, performances, *, alpha=1.0,
                        seed=42, n_bootstrap=10000, n_shifts=199):
    base_pred = grouped_predictions(base, y, groups, alpha)
    full_pred = grouped_predictions(np.column_stack([base, extra]), y, groups, alpha)
    base_loss = group_means(abs(y - base_pred), groups)
    full_loss = group_means(abs(y - full_pred), groups)
    effects = base_loss - full_loss
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shifts):
        shuffled = shift_within_performance(extra, performances, rng)
        pred = grouped_predictions(np.column_stack([base, shuffled]), y, groups, alpha)
        null.append(float(np.mean(base_loss - group_means(abs(y - pred), groups))))
    effect = float(effects.mean())
    summary = {
        "compositions": len(effects), "landmarks": len(y),
        "baseline_mae_log1p": float(base_loss.mean()),
        "ambiguity_mae_log1p": float(full_loss.mean()),
        "mae_improvement_log1p": effect,
        "relative_improvement_percent": float(100 * effect / base_loss.mean()),
        "improved_compositions": int((effects > 0).sum()),
        "bootstrap_ci95": bootstrap_mean(effects, seed, n_bootstrap),
        "shift_p_upper": float((1 + np.sum(np.asarray(null) >= effect)) / (1 + n_shifts)),
        "shift_effects": null,
    }
    return summary, base_pred, full_pred, effects
