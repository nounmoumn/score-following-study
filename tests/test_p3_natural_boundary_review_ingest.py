from scripts.p3_natural_boundary_review_ingest import DECISIONS, MISSED, compare_reviews


def test_review_vocabularies_are_disjoint_and_explicit():
    assert DECISIONS == {"accept", "reject", "uncertain"}
    assert MISSED == {"yes", "no", "uncertain"}
    assert "" not in DECISIONS | MISSED


def test_phase1_comparison_exposes_disagreement():
    base = {
        "candidates.csv": [{"candidate_id": "c1", "decision": "accept"}],
        "files.csv": [{"record_time": "f1", "missed_boundary_check": "no"}],
    }
    other = {
        "candidates.csv": [{"candidate_id": "c1", "decision": "reject"}],
        "files.csv": [{"record_time": "f1", "missed_boundary_check": "no"}],
    }
    disagreements = compare_reviews({"reviewer_A": base, "reviewer_B": other})
    assert disagreements == [{"file": "candidates.csv", "id": "c1", "reviewer_A": "accept", "reviewer_B": "reject"}]
