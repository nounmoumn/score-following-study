from scripts import p3_cause_specific_fhvb as analysis


def test_cause_set_is_fixed():
    assert analysis.CAUSES == ("observation_loss", "transition", "occurrence")
    assert analysis.PERMUTATIONS == 2_000
