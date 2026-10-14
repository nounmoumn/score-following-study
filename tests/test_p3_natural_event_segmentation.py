from scripts.p3_natural_event_restart_candidates import (
    event_anchors, event_restart_pairs, onset_events, unique_event_ngrams,
)


def test_onset_grouping_uses_upper_voice_token():
    notes = [(0.0, .2, 48), (.02, .3, 72), (.20, .4, 60)]
    events = onset_events(notes, .045)
    assert [e["token"] for e in events] == [72, 60]


def test_repeated_score_fingerprint_is_not_anchor():
    tokens = [60, 61, 62, 63, 60, 61, 62, 63, 70, 71, 72, 73]
    lookup = unique_event_ngrams(tokens)
    assert (60, 61, 62, 63) not in lookup
    assert lookup[(70, 71, 72, 73)] == 8


def test_synthetic_restart_recovered_and_monotone_rejected():
    score = list(range(40, 120))
    restarted = score[30:70] + score[5:45]
    anchors = event_anchors(restarted, score)
    found = event_restart_pairs(anchors, len(score))
    assert len(found) == 1
    assert 38 <= found[0]["performance_event_index"] <= 42
    assert event_restart_pairs(event_anchors(score, score), len(score)) == []
