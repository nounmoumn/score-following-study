import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import confidence_calibration_v1 as cal


def test_forbidden_columns_cover_every_label_derived_field():
    cfg = json.loads((ROOT / "configs/confidence_calibration_v1.json").read_text(encoding="utf-8"))
    used = {c for cols in cfg["feature_groups"].values() for c in cols}
    assert not used & cal.FORBIDDEN
    # The label and the quantity it is derived from must both be refused.
    assert {"correct", "error_score_sec", "truth_score_sec"} <= cal.FORBIDDEN


def test_matched_threshold_reaches_target_and_never_reads_labels():
    p = np.array([.1, .9, .5, .7, .3])
    w = np.full(5, .2)
    order = np.argsort(-p, kind="stable")
    threshold, selected = cal.matched_threshold(p, order, w, .6)
    assert selected.sum() == 3 and threshold == .5
    assert set(p[selected]) == {.5, .7, .9}
    # Lumpy weights make coverage overshoot: 96% of the weight sits on the
    # LOWEST-scoring row, so reaching .6 coverage forces every row in.
    heavy = np.array([.96, .01, .01, .01, .01])
    threshold, selected = cal.matched_threshold(p, order, heavy, .6)
    assert selected.all() and threshold == .1
    assert np.isclose(heavy[selected].sum()/heavy.sum(), 1.)
    # The target is a floor, never undershot.
    for target in (.05, .25, .5, .867, .95):
        _, sel = cal.matched_threshold(p, order, heavy, target)
        assert heavy[sel].sum()/heavy.sum() >= target - 1e-12


def test_selected_error_uses_weights_of_selected_rows_only():
    p = np.array([.9, .8, .2, .1])
    y = np.array([1., 0., 1., 1.])
    w = np.full(4, .25)
    order = np.argsort(-p, kind="stable")
    error, coverage = cal.selected_error(p, order, w, y, .5)
    assert np.isclose(coverage, .5) and np.isclose(error, .5)  # one of two selected is wrong


def test_work_weights_equalize_works_then_performances_then_observations():
    rows = ([{"work_family": "A", "case_id": "a"}]*8
            + [{"work_family": "A", "case_id": "b"}]*2
            + [{"work_family": "B", "case_id": "c"}])
    factor, index, families = cal.work_weight_factors(rows)
    weight = factor/len(families)
    weight = weight/weight.sum()
    np.testing.assert_allclose([weight[:8].sum(), weight[8:10].sum(), weight[10]], [.25, .25, .5])
    np.testing.assert_array_equal(index, [0]*10+[1])
    assert families == ["A", "B"]


def test_scoring_reproduces_a_known_reliability_and_selected_error():
    cfg = {"reliability_bins": 10, "matched_coverage": .5, "secondary_coverage": []}
    p = np.array([.95, .9, .1, .05])
    y = np.array([1., 0., 0., 0.])
    w = np.full(4, .25)
    out = cal.score_model(p, y, w, cfg, np.argsort(-p, kind="stable"))
    np.testing.assert_allclose(out["accuracy"], .25)
    np.testing.assert_allclose(out["brier"], np.mean([(.95-1)**2, .9**2, .1**2, .05**2]))
    assert np.isclose(out["selected_error_at_matched_coverage"], .5)
    assert np.isclose(out["achieved_matched_coverage"], .5)


def test_a_work_comes_from_the_score_profile_not_from_evaluated_rows():
    """Which frames are evaluated is set by annotation times, so a mean over
    evaluated rows would read ground-truth timing. A_work must not do that."""
    cfg = json.loads((ROOT / "configs/confidence_calibration_v1.json").read_text(encoding="utf-8"))
    profile_path = ROOT / cfg["score_profile"]
    if not profile_path.exists():
        pytest.skip("score profile not computed yet")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    audit = ROOT / cfg["audit_run"] / "primary_beats.csv"
    if not audit.exists():
        pytest.skip("audit run not present")
    rows, dropped, _ = cal.load_rows(cfg)
    values = {r["A_work"] for r in rows}
    assert values <= {v["mean"] for v in profile.values()}
    # Constant within a performance, and not equal to that performance's own row mean.
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    for case, group in by_case.items():
        assert len({g["A_work"] for g in group}) == 1
    differing = sum(1 for case, group in by_case.items()
                    if abs(group[0]["A_work"]
                           - np.mean([float(g["ambiguity"]) for g in group])) > 1e-9)
    assert differing > len(by_case)//2
    np.testing.assert_allclose([float(g["ambiguity"]) - g["A_work"] for g in rows[:50]],
                               [g["A_frame"] for g in rows[:50]], atol=1e-12)
