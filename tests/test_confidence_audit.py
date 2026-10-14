import numpy as np

from score_following.ambiguity import build_similarity_matrix, max_sim_ambiguity
from score_following.confidence_audit import (
    anchor_labels, blocked_maxsim, confidence_trace, summarize, work_weights,
)
from score_following.features import notes_to_chroma_stream
from score_following.p3_extension import bayes_from_cost


def test_completed_features_do_not_use_future_note_ends_or_notes():
    notes = [(0., 8., 60), (.15, .26, 64), (.99, 2., 67), (1.5, 3., 72)]
    prefix = [(s, min(e, 1.), p) for s,e,p in notes if s < 1.]
    full = notes_to_chroma_stream(notes, 5., .1)
    observed = notes_to_chroma_stream(prefix, 1., .1)
    np.testing.assert_allclose(full[:10], observed, atol=1e-14, rtol=0)


def test_blocked_maxsim_preserves_original_metric_and_undefined_tail():
    rng = np.random.default_rng(42)
    chroma = rng.random((70, 12))
    chroma[0:10] = 0
    sim, n = build_similarity_matrix(chroma, 5)
    expected = max_sim_ambiguity(sim, n, 8)
    actual, silent = blocked_maxsim(chroma, 5, 8, block=7)
    np.testing.assert_allclose(actual[:n], expected, atol=1e-12, rtol=0)
    assert np.isnan(actual[n:]).all() and silent == 6


def test_short_score_is_explicitly_missing():
    amb, silent = blocked_maxsim(np.ones((4, 12)), 5, 4)
    assert np.isnan(amb).all() and silent == 0


def test_trace_matches_original_and_has_prefix_invariance():
    rng = np.random.default_rng(13)
    cost = rng.random((60, 41))
    amb = np.linspace(0, 1, 41)
    full = confidence_trace(cost, amb)
    np.testing.assert_array_equal(full["map_index"], bayes_from_cost(cost))
    prefix = confidence_trace(cost[:17], amb)
    for field in full:
        np.testing.assert_array_equal(full[field][:17], prefix[field])


def test_sham_invariance_and_erasure_no_new_evidence():
    rng = np.random.default_rng(9)
    cost = rng.random((20, 25))
    amb = np.ones(25)
    original = confidence_trace(cost, amb)
    sham = confidence_trace(cost + np.arange(20)[:, None], amb)
    for field in original:
        np.testing.assert_allclose(original[field], sham[field], atol=1e-10, rtol=1e-10)
    erased = confidence_trace(np.ones((20, 25)), amb)
    np.testing.assert_allclose(erased["evidence"], 0, atol=1e-10)
    assert erased["confidence"][0] == 1.  # Certainty need not come from the new observation.


def test_missing_ambiguity_is_not_silently_dropped():
    trace = confidence_trace(np.zeros((10, 25)), np.full(25, np.nan))
    assert len(trace["confidence"]) == 10
    assert np.all(trace["ambiguity_valid_mass"] == 0)
    assert np.all(trace["ambiguity"] == 0)


def test_annotation_evaluation_uses_only_already_available_outputs():
    trace = {"map_index": np.array([0, 1, 2]), "position_score_sec": np.array([.1, .2, .3])}
    labels = anchor_labels(trace, [.05, .15, .25, .35], [.01, .1, .2, .4], tolerance=.01)
    np.testing.assert_array_equal(labels["indices"], [0, 1])
    np.testing.assert_array_equal(labels["correct"], [1, 1])
    np.testing.assert_allclose(labels["ages"], [.05, .05])


def test_labels_do_not_mutate_inference_and_clock_repair_is_forbidden():
    import pytest
    trace = confidence_trace(np.ones((10, 25)), np.zeros(25))
    copy = {k:v.copy() for k,v in trace.items()}
    anchor_labels(trace, [.15, .25], [0, 1])
    for key in trace:
        np.testing.assert_array_equal(trace[key], copy[key])
    with pytest.raises(ValueError):
        anchor_labels(trace, [.25, .15], [0, 1])


def test_work_weighting_is_not_frame_or_performance_weighting():
    rows = [{"work_family":"A", "case_id":"a"}]*8
    rows += [{"work_family":"A", "case_id":"b"}]*2
    rows += [{"work_family":"B", "case_id":"c"}]
    w = work_weights(rows)
    np.testing.assert_allclose([w[:8].sum(),w[8:10].sum(),w[10]], [.25,.25,.5])


def test_reliability_at_confidence_one_and_empty_selected_group():
    rows = [{"confidence":1., "correct":0}, {"confidence":1., "correct":1}]
    stats = summarize(rows)
    assert stats["brier"] == .5 and stats["ece"] == .5
    assert stats["high_confidence_accuracy"] == .5
    assert stats["reliability"][-1]["bin"] == 9
    low = summarize([{"confidence":.2, "correct":0}])
    assert low["high_confidence_accuracy"] is None


# --- e saturation repair (2026-09-22). v1 kept logit(mass); once the filter
# concentrated, both masses clipped and evidence returned exactly 0 on 13.0% of
# real observations. These pin the exact-log-odds replacement. ---

def test_exact_log_odds_agrees_with_v1_logit_away_from_saturation():
    from score_following.confidence_audit import logit, regional_log_odds
    rng = np.random.default_rng(3)
    prior = rng.random(50) + .05
    prior /= prior.sum()
    likelihood = rng.random(50) + .05
    unnormalized = prior * likelihood
    region = np.zeros(50, bool)
    region[20:31] = True
    odds_c, odds_u, saturated = regional_log_odds(prior, unnormalized, region)
    posterior = unnormalized / unnormalized.sum()
    assert saturated == 0
    np.testing.assert_allclose(odds_c, logit(posterior[region].sum()), rtol=1e-12)
    np.testing.assert_allclose(odds_u, logit(prior[region].sum()), rtol=1e-12)


def test_concentrated_filter_no_longer_reports_zero_evidence():
    from score_following.confidence_audit import logit, regional_log_odds
    prior = np.full(200, 1e-18)
    prior[95:106] = 1./11
    prior /= prior.sum()
    unnormalized = prior * np.where(np.arange(200) == 100, 1., 1e-6)
    region = np.zeros(200, bool)
    region[95:106] = True
    odds_c, odds_u, saturated = regional_log_odds(prior, unnormalized, region)
    assert saturated == 0
    # v1: both masses clip to 1-1e-12, so the difference collapses to zero.
    assert logit(prior[region].sum()) == logit((unnormalized/unnormalized.sum())[region].sum())
    assert odds_c - odds_u > 1.  # the observation did concentrate inside the set
    assert odds_c > 30 and odds_u > 30  # both beyond the v1 clip bound of ~27.6


def test_log_odds_stay_finite_and_resolve_where_mass_is_compressed():
    """Clipping never triggered on the real v1 cohort (min 1-c was 2.6e-9).

    The repair is therefore not a fix for observed zeros; it removes a latent
    failure mode and gives calibration inputs a scale that still resolves when
    61.7% of confidences sit above .999. Beta here is raised past the audit
    setting purely to drive the mass where logit() would stop resolving.
    """
    n = 240
    cost = np.ones((60, n))
    for t in range(60):
        cost[t, t+1] = 0.
    trace = confidence_trace(cost, np.zeros(n), beta=60., restart=1e-12)
    late = slice(20, 60)
    assert np.all(trace["confidence"][late] == 1.)      # logit() resolves nothing here
    assert np.all(trace["evidence_saturated"][late] == 0)
    assert np.all(trace["confidence_log_odds"][late] > 30.)
    assert np.all(np.isfinite(trace["confidence_log_odds"]))
    # Two genuinely different concentrations both read as confidence == 1.0,
    # yet remain distinguishable in log odds. That is the resolution being bought.
    sharper = confidence_trace(cost, np.zeros(n), beta=120., restart=1e-12)
    assert np.all(sharper["confidence"][late] == 1.)
    assert sharper["confidence_log_odds"][late].mean() > trace["confidence_log_odds"][late].mean() + 10.


def test_flat_likelihood_from_performance_silence_reports_exactly_zero_evidence():
    """cosine_cost gives a constant row for an all-zero performance frame, so the
    likelihood is flat and e=0 is the correct value, not a numerical artifact.
    660 of 5058 v1 beats (13.0%) were this case; none were clipped."""
    n = 40
    cost = np.concatenate([np.tile(np.arange(n, dtype=float)/n, (10, 1)),
                           np.full((5, n), 2.)])
    trace = confidence_trace(cost, np.zeros(n))
    silent = slice(10, 15)
    np.testing.assert_allclose(trace["evidence"][silent], 0, atol=1e-12)
    np.testing.assert_allclose(trace["confidence"][silent],
                               trace["prior_confidence"][silent], atol=1e-15)
    assert np.all(trace["evidence_saturated"][silent] == 0)


def test_structural_degeneracy_is_flagged_not_reported_as_zero_evidence():
    # Frame 0 has a one-hot prior: no mass outside the set, odds infinite.
    trace = confidence_trace(np.zeros((5, 25)), np.zeros(25))
    assert trace["evidence_saturated"][0] == 1
    # A set covering every state has an empty complement.
    full = confidence_trace(np.zeros((5, 8)), np.zeros(8), tolerance=5.)
    assert np.all(full["evidence_saturated"] == 1) and np.all(full["evidence"] == 0)


def test_repair_preserves_path_sham_and_erasure_identities():
    rng = np.random.default_rng(21)
    cost = rng.random((40, 60))
    amb = np.linspace(0, 1, 60)
    trace = confidence_trace(cost, amb)
    np.testing.assert_array_equal(trace["map_index"], bayes_from_cost(cost))
    sham = confidence_trace(cost + np.arange(40)[:, None], amb)
    for field in trace:
        np.testing.assert_allclose(trace[field], sham[field], atol=1e-10, rtol=1e-10)
    flat = confidence_trace(np.ones((20, 25)), amb[:25])
    np.testing.assert_allclose(flat["evidence"], 0, atol=1e-12)
    prefix = confidence_trace(cost[:11], amb)
    for field in trace:
        np.testing.assert_array_equal(trace[field][:11], prefix[field])


def test_flat_likelihood_flag_marks_exactly_the_constant_cost_rows():
    n = 30
    cost = np.tile(np.arange(n, dtype=float)/n, (8, 1))
    cost[3] = 2.        # zero-energy performance frame
    cost[6] = 0.        # degenerate but still constant
    trace = confidence_trace(cost, np.zeros(n))
    np.testing.assert_array_equal(trace["flat_likelihood"], [0, 0, 0, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(trace["evidence"][[3, 6]], 0, atol=1e-12)
