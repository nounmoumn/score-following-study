"""Conservative error/event bounds. Caller must distinguish certified GT from scenarios."""
import numpy as np

def error_bounds(position, lower, upper):
    x, lo, hi = np.broadcast_arrays(np.asarray(position, float), np.asarray(lower, float), np.asarray(upper, float))
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo > hi):
        raise ValueError('Finite ordered bounds required; unknown must be handled explicitly')
    return np.maximum(np.maximum(lo-x, x-hi), 0.), np.maximum(abs(x-lo), abs(x-hi))

def sustained_bounds(error_lower, error_upper, start, stop, dwell=5, threshold=.5):
    """Onset in [start,stop); confirmation can extend dwell-1 frames past stop.

    A possible run is an outer bound, not proof of a jointly feasible GT path.
    """
    lo, hi = np.asarray(error_lower), np.asarray(error_upper)
    if lo.ndim != 1 or hi.shape != lo.shape or start < 0 or stop <= start or dwell < 1 or stop+dwell-1 > len(lo):
        raise ValueError('Invalid window or incomplete confirmation support')
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo < 0) or np.any(lo > hi):
        raise ValueError('Invalid error bounds')
    definite = [i for i in range(start,stop) if np.all(lo[i:i+dwell] > threshold)]
    possible = [i for i in range(start,stop) if np.all(hi[i:i+dwell] > threshold)]
    return {'definite_event':bool(definite), 'possible_event':bool(possible),
            'earliest_possible_onset':possible[0] if possible else None,
            'latest_guaranteed_first_onset':definite[0] if definite else None}
