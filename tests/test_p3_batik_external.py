import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "results/p3_batik_external" / name).read_text(encoding="utf-8"))


def test_manifest_keeps_development_and_overlap_families_out():
    manifest = load("manifest.json")
    assert manifest["eligible_movements"] == 21
    assert not ({"K.279", "K.331", "K.332"} & set(manifest["eligible_sonatas"]))
    assert all(row["duration_sec"] >= 90 for row in manifest["rows"] if row["eligible"])
    assert all(row["clock_nonpositive_steps"] == 0 for row in manifest["rows"] if row["eligible"])


def test_result_confusion_and_decision_recompute():
    result = load("result.json")
    rows = result["rows"]
    for field, key in [("two_consecutive", "primary_two_consecutive"),
                       ("any_positive", "secondary_any_positive")]:
        expected = {
            "tp": sum(r[field] and r["sustained"] for r in rows),
            "fp": sum(r[field] and not r["sustained"] for r in rows),
            "fn": sum(not r[field] and r["sustained"] for r in rows),
            "tn": sum(not r[field] and not r["sustained"] for r in rows),
        }
        assert {k: result[key][k] for k in expected} == expected
    assert len({(r["sonata"], r["movement"], r["onset_sec"])
                for r in rows if r["sustained"]}) == result["unique_event_onsets"] == 5
    assert result["decision"] == "negative"
    assert result["max_bayes_identity_residual"] < 1e-12


def test_warning_availability_never_uses_event_confirmation():
    for row in load("result.json")["rows"]:
        if row["warning_available_sec"] is not None:
            assert row["warning_available_sec"] == row["time_sec"]
        if row["sustained"]:
            assert math.isclose(row["confirmation_sec"], row["onset_sec"] + .4,
                                rel_tol=0, abs_tol=1e-12)
