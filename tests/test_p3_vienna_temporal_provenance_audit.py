from scripts import p3_vienna_temporal_provenance_audit as audit


def test_preexisting_script_list_excludes_future_validation():
    assert "p3_vienna_observation_shock_validation.py" not in audit.PREEXISTING
    assert len(audit.PREEXISTING) == 8
