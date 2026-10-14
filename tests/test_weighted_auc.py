"""The AUC used by the smoothing runner and the calibration diagnostic must be the
standard weighted one, including tie handling, or the ordering claims are unfounded."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOOTHING = load("confidence_smoothing_v1")
DIAGNOSTIC = load("confidence_calibration_v1_diagnostic")


def test_matches_sklearn_with_sample_weights():
    rng = np.random.default_rng(7)
    for _ in range(5):
        n = rng.integers(40, 300)
        score = rng.random(n)
        y = (rng.random(n) < .7).astype(float)
        w = rng.random(n) + .01
        expected = roc_auc_score(y, score, sample_weight=w)
        assert np.isclose(SMOOTHING.weighted_auc(score, y, w), expected, atol=1e-10)
        assert np.isclose(DIAGNOSTIC.weighted_auc(score, y, w), expected, atol=1e-10)


def test_ties_are_counted_at_one_half():
    # Every score identical: no ordering information at all.
    score = np.ones(20)
    y = np.array([1.]*12 + [0.]*8)
    w = np.full(20, .05)
    assert np.isclose(SMOOTHING.weighted_auc(score, y, w), .5)
    # A single tied pair between otherwise perfect groups.
    score = np.array([.9, .5, .5, .1])
    y = np.array([1., 1., 0., 0.])
    w = np.full(4, .25)
    expected = roc_auc_score(y, score, sample_weight=w)
    assert np.isclose(SMOOTHING.weighted_auc(score, y, w), expected, atol=1e-12)


def test_perfect_and_inverted_orderings():
    score = np.arange(10, dtype=float)
    y = np.array([0.]*5 + [1.]*5)
    w = np.full(10, .1)
    assert np.isclose(SMOOTHING.weighted_auc(score, y, w), 1.)
    assert np.isclose(SMOOTHING.weighted_auc(-score, y, w), 0.)


def test_single_class_returns_none():
    score = np.array([.1, .2, .3])
    w = np.full(3, 1/3)
    assert SMOOTHING.weighted_auc(score, np.ones(3), w) is None
    assert SMOOTHING.weighted_auc(score, np.zeros(3), w) is None
