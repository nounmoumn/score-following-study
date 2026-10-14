import json

from scripts import p3_os_provenance_audit as audit


def test_audit_declares_no_independent_validation():
    source = audit.ROOT / "scripts/p3_os_provenance_audit.py"
    assert source.exists()
    stored = json.loads((audit.ROOT / "results/p3_observation_shock_validation/result.json").read_text())
    assert stored["conditional_p3_prime_confirmed"] is False
    assert stored["gates"]["validation_outcomes_previously_unopened"] is False
