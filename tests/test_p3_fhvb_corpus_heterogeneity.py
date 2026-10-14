from scripts import p3_fhvb_corpus_heterogeneity as audit


def test_pooled_auc_is_half_for_tied_values():
    rows = [
        {"event": True, "deficit": 1.0},
        {"event": False, "deficit": 1.0},
    ]
    assert audit.pooled_auc(rows) == 0.5


def test_corpus_summary_keeps_eventless_tracks_in_track_count():
    rows = [
        {"corpus": "x", "track_key": "a", "event": True, "deficit": 1.0, "onset_sec": 1.0},
        {"corpus": "x", "track_key": "a", "event": False, "deficit": 0.0, "onset_sec": None},
        {"corpus": "x", "track_key": "b", "event": False, "deficit": 0.0, "onset_sec": None},
    ]
    result = audit.corpus_summary(rows, "x")
    assert result["tracks"] == 2
    assert result["event_tracks_with_defined_auc"] == 1
