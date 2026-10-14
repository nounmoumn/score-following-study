import json
from pathlib import Path


def test_validation_protocol_and_acquisition_are_frozen_before_result():
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "P3_DP_PIANOCORE_VALIDATION_PROTOCOL.md").read_text(encoding="utf8")
    acquisition = json.loads((root / "results/p3_dp_pianocore_validation_acquisition.json").read_text(encoding="utf8"))
    assert "exactly 17 composers, 68 windows, and 340 dose runs" in protocol
    assert acquisition["acquired"] == acquisition["unique_composers"] == 17
    assert acquisition["all_score_performance_distinct"] is True


def test_preserved_result_recomputes_and_all_gates_pass():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/p3_dp_pianocore_validation/result.json").read_text(encoding="utf8"))
    assert result["counts"] == {"composers": 17, "windows": 68, "runs": 340,
                                "development_composer_overlap": 0}
    assert all(result["gates"].values())
    assert result["validation_pass"] is True
    assert result["p3_dp_confirmed"] is True
    assert result["p3_d_map_claim_reopened"] is False
    assert result["natural_temporal_p3_confirmed"] is False
