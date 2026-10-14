import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads((ROOT / path).read_text(encoding="utf-8"))


def auc(positive, negative):
    return sum(1 if p > n else .5 if p == n else 0 for p in positive for n in negative) / (len(positive) * len(negative))


def test_batik_new_signal_gate_recomputes():
    result = read("results/p3_coherent_trajectory/result.json"); rows = result["rows"]
    tp = sum(r["warning"] and r["sustained"] for r in rows)
    fp = sum(r["warning"] and not r["sustained"] for r in rows)
    fn = sum(not r["warning"] and r["sustained"] for r in rows)
    tn = sum(not r["warning"] and not r["sustained"] for r in rows)
    assert (tp, fp, fn, tn) == (3, 41, 2, 98)
    positive = [r["C"] for r in rows if r["sustained"]]
    negative = [r["C"] for r in rows if not r["sustained"]]
    assert math.isclose(auc(positive, negative), result["C_rank_auc_diagnostic"])
    assert result["gate_pass"] is False


def test_kraisler_manifest_and_transfer_gate_recompute():
    manifest = read("results/p3_kraisler_signal/manifest.json")
    assert manifest["eligible"] == 13
    assert all(r["duration_sec"] >= 45 and r["clock_nonpositive_steps"] == 0
               for r in manifest["rows"] if r["eligible"])
    result = read("results/p3_kraisler_signal/result.json"); rows = result["rows"]
    assert len({(r["track"], r["onset_sec"]) for r in rows if r["sustained"]}) == 11
    positive = [r["C"] for r in rows if r["sustained"]]
    negative = [r["C"] for r in rows if not r["sustained"]]
    assert math.isclose(auc(positive, negative), result["landmark_rank_auc"])
    assert result["landmark_rank_auc"] < .7
    assert result["gate_pass"] is False and result["decision"] == "stop_signal_family"

