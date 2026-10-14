from scripts.p3_natural_unit_review_finalize import compare


def test_phase2_conflict_is_not_silently_resolved():
    a = [{"unit_id": "u1", "decision": "accept"}]
    b = [{"unit_id": "u1", "decision": "uncertain"}]
    assert compare(a, b) == [{"unit_id": "u1", "reviewer_A": "accept", "reviewer_B": "uncertain"}]
