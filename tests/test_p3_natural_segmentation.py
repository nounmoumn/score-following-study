from scripts.p3_natural_restart_candidates import (
    exact_anchors, merge_candidates, restart_pairs, unique_score_ngrams,
)


def test_unique_score_ngrams_excludes_repeated_material():
    score = [60, 61, 62, 63, 64, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74]
    lookup = unique_score_ngrams(score)
    assert (60, 61, 62, 63, 64) not in lookup
    assert lookup[(70, 71, 72, 73, 74)] == 10


def test_restart_requires_supported_large_backward_jump():
    # Three locally supporting anchors on either side of a 500-note restart.
    anchors = [
        {"performance_index": 0, "score_index": 600},
        {"performance_index": 5, "score_index": 605},
        {"performance_index": 10, "score_index": 610},
        {"performance_index": 15, "score_index": 100},
        {"performance_index": 20, "score_index": 105},
        {"performance_index": 25, "score_index": 110},
    ]
    found = restart_pairs(anchors, 1000)
    assert len(found) == 1
    assert found[0]["performance_index"] == 15


def test_exact_anchors_and_candidate_merge_are_deterministic():
    score = list(range(60, 75))
    anchors = exact_anchors(score[:8] + score[3:12], score)
    assert anchors[0] == {"performance_index": 0, "score_index": 0}
    merged = merge_candidates([
        {"time_sec": 10.0, "performance_index": 10, "signals": ["hard_pause"], "members": [{}]},
        {"time_sec": 12.0, "performance_index": 12, "signals": ["content_restart"], "members": [{}]},
    ])
    assert len(merged) == 1
    assert merged[0]["signals"] == ["content_restart", "hard_pause"]
    assert merged[0]["time_sec"] == 12.0
