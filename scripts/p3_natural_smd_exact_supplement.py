#!/usr/bin/env python3
"""Preflight every prospectively fixed exact SMD/ASAP supplement pair."""

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.p3_natural_restart_candidates import load_notes
except ImportError:
    from p3_natural_restart_candidates import load_notes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_natural_smd_exact_supplement.json"
PAIRS = [
    ("SMD_Haydn_Sonata52_mvt1", "Haydn,_Joseph", "Piano Sonata No.52 Mvt.1", "HAYDN_HOB_XVI_39_1",
     "data/smd_direct_midi/Haydn_HobXVINo52-01_008_20110315-SMD.mid", "asap_dataset/Haydn/Keyboard_Sonatas/39-1/midi_score.mid"),
    ("SMD_Haydn_Sonata52_mvt2", "Haydn,_Joseph", "Piano Sonata No.52 Mvt.2", "HAYDN_HOB_XVI_39_2",
     "data/smd_direct_midi/Haydn_HobXVINo52-02_008_20110315-SMD.mid", "asap_dataset/Haydn/Keyboard_Sonatas/39-2/midi_score.mid"),
    ("SMD_Haydn_Sonata52_mvt3", "Haydn,_Joseph", "Piano Sonata No.52 Mvt.3", "HAYDN_HOB_XVI_39_3",
     "data/smd_direct_midi/Haydn_HobXVINo52-03_008_20110315-SMD.mid", "asap_dataset/Haydn/Keyboard_Sonatas/39-3/midi_score.mid"),
    ("SMD_Liszt_LaLeggierezza", "Liszt,_Franz", "Concert Etude S.145 No.2 La leggierezza", "LISZT_S145_2",
     "data/smd_direct_midi/Liszt_KonzertetuedeNo2LaLeggierezza_003_20090916-SMD.mid", "asap_dataset/Liszt/Concert_Etude_S145/2/midi_score.mid"),
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    x = np.bincount([note[2] % 12 for note in a], minlength=12).astype(float)
    y = np.bincount([note[2] % 12 for note in b], minlength=12).astype(float)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def main():
    rows = []
    for record_time, composer, piece, work_id, performance_rel, score_rel in PAIRS:
        performance_path, score_path = ROOT/performance_rel, ROOT/score_rel
        performance, score = load_notes(performance_path), load_notes(score_path)
        duration = max(n[1] for n in performance); similarity = cosine(performance, score)
        gates = {"duration_ge_90": duration >= 90, "performance_notes_ge_200": len(performance) >= 200,
                 "score_notes_ge_100": len(score) >= 100, "pitch_class_cosine_ge_0_80": similarity >= .80,
                 "performance_score_hash_distinct": sha(performance_path) != sha(score_path)}
        rows.append({"record_time": record_time, "score_composer": composer, "piece": piece,
            "rule_index": work_id, "performance_path": performance_rel, "score_midi_path": score_rel,
            "performance_duration_sec": duration, "score_duration_sec": max(n[1] for n in score),
            "performance_notes": len(performance), "score_notes": len(score),
            "pitch_class_cosine": similarity, "performance_sha256": sha(performance_path),
            "score_sha256": sha(score_path), "gates": gates, "eligible": all(gates.values())})
    result = {"scope": "Outcome-blind exact-catalogue SMD supplement; no tracker/P3 outcomes",
              "counts": {"fixed_pairs": len(rows), "eligible": sum(r["eligible"] for r in rows),
                         "composers": len({r["score_composer"] for r in rows if r["eligible"]})},
              "all_fixed_pairs_eligible": all(r["eligible"] for r in rows), "validation_eligible": False, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
