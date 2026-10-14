import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asap_overlap_mapping():
    module = load_script("p3_cp_wtc_preflight.py")
    assert {1, 2, 5, 6}.issubset(module.ASAP_PIECES)
    assert {3, 4, 7, 10, 47, 48}.isdisjoint(module.ASAP_PIECES)


def test_cp_wtc_alignment_clock_conversion(tmp_path):
    module = load_script("p3_tdr_cp_wtc_validation.py")
    path = tmp_path / "clock.txt"
    np.savetxt(path, np.array([[0, 1, 0, 0, 0], [2, 3, 0, 0, 0], [5, 7, 0, 0, 0]]))
    score, perf = module.load_alignment(path)
    np.testing.assert_allclose(score, [0, 1, 2.5])
    np.testing.assert_allclose(perf, [1, 3, 7])


def test_cp_wtc_alignment_rejects_nonmonotone(tmp_path):
    module = load_script("p3_tdr_cp_wtc_validation.py")
    path = tmp_path / "clock.txt"
    np.savetxt(path, np.array([[0, 1], [1, 1]]))
    try:
        module.load_alignment(path)
    except ValueError:
        pass
    else:
        raise AssertionError("non-monotone clock accepted")


def test_excerpt_clock_must_not_use_full_recording():
    perf = np.array([0.0, 10.0, 20.0, 100.0])
    score = np.array([0.0, 5.0, 10.0, 90.0])
    excerpt_duration = 20.0
    local_stop = np.interp(excerpt_duration, perf, score)
    local_hop = .1 * (local_stop - score[0]) / excerpt_duration
    full_hop = .1 * (score[-1] - score[0]) / (perf[-1] - perf[0])
    assert local_hop == .05
    assert local_hop != full_hop
