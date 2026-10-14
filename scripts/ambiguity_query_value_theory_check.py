"""Exact finite probability checks, NOT performance data or research validation.

Expected Bayes error reduction is a standard decision-theoretic quantity.
These checks expose prerequisites rather than claim a new selection method.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def risk(probabilities, losses):
    """Loss matrix: actions x hypotheses. Actions may ignore an answer."""
    return float(np.min(losses @ probabilities))


def value(probabilities, losses, channel):
    """channel[h, answer]; sum unnormalised conditional risks exactly."""
    p, loss, q = map(np.asarray, (probabilities, losses, channel))
    assert np.all(p >= 0) and np.isclose(p.sum(), 1)
    assert np.all(q >= 0) and np.allclose(q.sum(axis=1), 1)
    return risk(p, loss) - sum(risk(p * q[:, a], loss) for a in range(q.shape[1]))


def exact_channel(labels):
    return np.eye(max(labels) + 1)[labels]


def entropy(probabilities):
    p = np.asarray(probabilities)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    # Ten evaluation positions follow one latent branch. An independent query
    # is maximally uncertain but cannot improve those positions.
    p = np.array([.4, .4, .1, .1])
    branch = np.array([0, 0, 1, 1])
    independent = np.array([0, 1, 0, 1])
    losses = 10 * (np.arange(2)[:, None] != branch[None, :])
    irrelevant = value(p, losses, exact_channel(independent))
    relevant = value(p, losses, exact_channel(branch))
    assert np.isclose(irrelevant, 0) and np.isclose(relevant, 2)
    # Identical query and target marginals, different temporal coupling.
    correlated = value(np.array([.5, .5]), np.array([[0., 10.], [10., 0.]]), np.eye(2))
    independent_value = value(np.full(4, .25), losses, exact_channel(independent))
    assert np.isclose(correlated, 5) and np.isclose(independent_value, 0)
    # Two exhaustive branches, disagreement L, symmetric answer error e <= .5.
    # Exact Bayes value = L * max(0, min(p, 1-p) - e).
    errors = []
    for prior in (.01, .1, .3, .5, .8, .99):
        for epsilon in (0., .05, .2, .5):
            for length in (1., 10., 100.):
                got = value(np.array([prior, 1-prior]),
                            np.array([[0., length], [length, 0.]]),
                            np.array([[1-epsilon, epsilon], [epsilon, 1-epsilon]]))
                expected = length * max(0., min(prior, 1-prior)-epsilon)
                errors.append(abs(got-expected))
    assert max(errors) < 1e-12
    # A confident wrong model predicts no benefit, while a true correction helps.
    confident_gain = value(np.array([1., 0.]), np.array([[0., 10.], [10., 0.]]), np.eye(2))
    assert confident_gain == 0
    result = {
        "scope": "algebra counterexamples only; no corpus, no empirical utility claim",
        "uncertain_irrelevant_query": {"entropy_bits": entropy([.5,.5]), "expected_errors_removed": irrelevant},
        "less_uncertain_relevant_query": {"entropy_bits": entropy([.8,.2]), "expected_errors_removed": relevant},
        "same_marginals_different_coupling": {"independent": independent_value, "correlated": correlated},
        "binary_formula_grid_cases": len(errors), "max_absolute_error": max(errors),
        "confident_wrong_model_predicted_gain": confident_gain,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": "PASS"
    }
    out = Path(__file__).resolve().parents[1] / "results/ambiguity_query_value_theory_check.json"
    if out.exists():
        previous = json.loads(out.read_text())
        assert previous == result, "Existing result differs; preserve it and investigate"
    else:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
