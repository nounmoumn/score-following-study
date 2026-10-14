import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_pt_kraisler_transfer.py"
SPEC = importlib.util.spec_from_file_location("p3_pt_kraisler_transfer", SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def test_aggregate_keeps_risk_and_incidence_endpoints_only():
    rows = [{"track": "a", "risk_effect": .1, "erased_event": 1, "intact_event": 0},
            {"track": "a", "risk_effect": .3, "erased_event": 0, "intact_event": 0}]
    result = transfer.aggregate(rows)[0]
    assert result["risk_effect"] == .2
    assert result["incidence_effect"] == .5
    assert "map_effect" not in result

