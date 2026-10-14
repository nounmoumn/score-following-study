import numpy as np

from score_following.evidence_ablation import CostMatrixDTW
from score_following.p3_extension import (oltw_from_cost, bayes_from_cost, cosine_cost,
    distractor_costs, cue_costs, evidence_at, score_window_features,
    score_ambiguity_at, holm_adjust)
from score_following.ambiguity import build_similarity_matrix, max_sim_ambiguity


def test_compiled_oltw_matches_original_with_ties_and_random_features():
    rng = np.random.default_rng(23)
    for discrete in [False, True]:
        for m, n, band in [(45, 70, 8), (70, 45, 8), (45, 45, 15)]:
            u = np.eye(12)[rng.integers(12, size=m)] if discrete else rng.random((m, 12))
            v = np.eye(12)[rng.integers(12, size=n)] if discrete else rng.random((n, 12))
            original = CostMatrixDTW(u, v, band_width=band)
            original.run(stop_at_performance_end=True)
            expected = np.zeros(m, dtype=int)
            for t, j in original.real_path():
                expected[t] = j
            np.testing.assert_array_equal(oltw_from_cost(original.costs, band), expected)


def test_bayes_is_causal_and_tracks_unique_evidence():
    cost = np.ones((40, 60))
    cost[np.arange(40), np.arange(40)] = 0
    full = bayes_from_cost(cost)
    np.testing.assert_array_equal(full, np.arange(40))
    np.testing.assert_array_equal(full[:20], bayes_from_cost(cost[:20]))
    cost[20:] = 99
    np.testing.assert_array_equal(full[:20], bayes_from_cost(cost)[:20])


def test_distractor_preserves_truth_and_has_matched_cost_budget():
    rng = np.random.default_rng(24)
    base = rng.random((60, 160))
    truth = np.arange(60)+40
    for i, j in enumerate(truth):
        base[i, j-2:j+3] = .1
    target, sham, info = distractor_costs(base, truth, onset=20, end=40,
        offset=-12, guard=2, radius=40, seed=24)
    assert info['total_reduction'] > 0
    np.testing.assert_array_equal(target[:20], base[:20])
    np.testing.assert_array_equal(target[40:], base[40:])
    np.testing.assert_allclose(target.sum(axis=1), sham.sum(axis=1))
    np.testing.assert_allclose(abs(target-base).sum(axis=1), abs(sham-base).sum(axis=1))
    for i, j in enumerate(truth):
        np.testing.assert_array_equal(target[i, j-2:j+3], base[i, j-2:j+3])
        np.testing.assert_array_equal(sham[i, j-2:j+3], base[i, j-2:j+3])


def test_cue_and_sham_have_same_average_cost_but_only_cue_is_informative():
    base = np.ones((50, 100))
    truth = np.arange(50)+10
    cue = cue_costs(base, truth, first=20, end=30, guard=2, strength=.25)
    sham = cue_costs(base, truth, first=20, end=30, guard=2, strength=.25, sham=True)
    np.testing.assert_allclose(cue.mean(axis=1), sham.mean(axis=1))
    np.testing.assert_array_equal(cue[:20], base[:20])
    assert cue[25, int(truth[25])] == 1
    assert np.ptp(sham[25]) == 0
    assert np.ptp(cue[25]) > 0


def test_evidence_does_not_read_future_live_input():
    rng = np.random.default_rng(7)
    v = rng.random((100, 12))
    u = v[:60].copy()
    truth = np.arange(60)
    before = evidence_at(u, v, truth, 30, window=10, exclude=3, radius=20, frontier=35, band=25)
    u[31:] = 1000
    after = evidence_at(u, v, truth, 30, window=10, exclude=3, radius=20, frontier=35, band=25)
    assert before == after
    assert before['weakness_local'] < 0


def test_landmark_ambiguity_is_same_maxsim_with_completed_timestamp():
    rng = np.random.default_rng(44)
    v = rng.random((100, 12))
    sim, n = build_similarity_matrix(v, 10)
    reference = max_sim_ambiguity(sim, n, 5)
    features = score_window_features(v, 10)
    for end in [20, 30, 60, 90]:
        np.testing.assert_allclose(score_ambiguity_at(features, end, 10, 5), reference[end-10])


def test_holm_is_monotone_in_rank_and_preserves_original_order():
    np.testing.assert_allclose(holm_adjust([.04, .001, .02]), [.04, .003, .04])


def test_optional_audit_reads_original_accumulated_cost_without_changing_path():
    rng = np.random.default_rng(81)
    u, v = rng.random((60, 12)), rng.random((90, 12))
    candidates = np.full((60, 2), -1, dtype=np.int64)
    candidates[20] = [18, 22]
    candidates[40] = [35, 38]
    class RecordedDTW(CostMatrixDTW):
        def _get_inc(self):
            t, j = self.t, self.j
            vals = []
            for candidate in candidates[t]:
                vals.append(self._norm(t, candidate) if max(0, j-self.C+1)<=candidate<=j
                            and self.L.get((t, candidate), 0)>0 else np.nan)
            self.snapshots[t] = vals + [j]
            return super()._get_inc()
    original = RecordedDTW(u, v, band_width=12)
    original.snapshots = {}
    original.run(stop_at_performance_end=True)
    values = np.full((60, 3), np.nan)
    audited = oltw_from_cost(original.costs, 12, audit_candidates=candidates, audit_values=values)
    np.testing.assert_array_equal(audited, oltw_from_cost(original.costs, 12))
    for i in [20, 40]:
        np.testing.assert_allclose(values[i], original.snapshots[i], equal_nan=True)
