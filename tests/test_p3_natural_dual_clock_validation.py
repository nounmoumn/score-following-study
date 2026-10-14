import numpy as np

from scripts import p3_natural_dual_clock_validation as selector


def performance(n, spacing=1.0):
    return [{"time_sec": i * spacing, "latest_end_sec": i * spacing + .1, "pitches": {60}} for i in range(n)]


def test_selector_averages_two_usable_clocks():
    exact = np.asarray([0, 1, 2, 3])
    chroma = np.asarray([0, 1, 2, 3])
    selected, branch, disagreement = selector.select_clock(exact, chroma, np.asarray([0.0, .5, 1.0, 1.5]), performance(4))
    assert branch == "mean_exact_chroma"
    assert disagreement == 0.0
    assert np.allclose(selected, [0.0, .5, 1.0, 1.5])


def test_selector_uses_chroma_when_exact_has_long_plateau():
    exact = np.asarray([0, 0, 0, 0])
    chroma = np.asarray([0, 1, 2, 3])
    selected, branch, _ = selector.select_clock(exact, chroma, np.asarray([0.0, .5, 1.0, 1.5]), performance(4, spacing=2.0))
    assert branch == "chroma_only"
    assert np.allclose(selected, [0.0, .5, 1.0, 1.5])
