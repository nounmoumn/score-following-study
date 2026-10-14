#!/usr/bin/env python3
"""File-only eligibility and practice-restart audit for PianoVAM P3 material."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p3_pianovam_acquisition.json"
OUT = ROOT / "results/p3_pianovam_file_preflight.json"


def notes(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    values = sorted((n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes)
    return values, float(pm.get_end_time())


def cosine_pitch_class(a, b):
    x = np.bincount([n[2] % 12 for n in a], minlength=12).astype(float)
    y = np.bincount([n[2] % 12 for n in b], minlength=12).astype(float)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) if x.any() and y.any() else 0.


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8")); cache = {}; rows = []
    for row in source["rows"]:
        perf, duration = notes(ROOT / row["performance_path"])
        if row["score_midi_path"] not in cache:
            cache[row["score_midi_path"]] = notes(ROOT / row["score_midi_path"])
        score, score_duration = cache[row["score_midi_path"]]
        gaps = [perf[i+1][0] - perf[i][1] for i in range(len(perf)-1) if perf[i+1][0] - perf[i][1] >= 5.]
        similarity = cosine_pitch_class(perf, score)
        gates = {"duration_ge_90": duration >= 90., "performance_notes_ge_200": len(perf) >= 200,
                 "score_notes_ge_100": len(score) >= 100, "pitch_class_cosine_ge_0_80": similarity >= .80}
        rows.append({"record_time": row["record_time"], "composer": row["composer"], "piece": row["piece"],
                     "score_composer": row["score_composer"], "rule_index": row["rule_index"],
                     "performance_duration_sec": duration, "score_duration_sec": score_duration,
                     "performance_notes": len(perf), "score_notes": len(score),
                     "pitch_class_cosine": similarity, "gaps_ge_5sec": len(gaps),
                     "maximum_gap_sec": max(gaps, default=0.), "gates": gates,
                     "file_eligible": all(gates.values())})
    good = [row for row in rows if row["file_eligible"]]
    works = {row["rule_index"] for row in good}; composers = {row["score_composer"] for row in good}
    cohort_gates = {"direct_midi_ge_20": len(good) >= 20, "work_families_ge_10": len(works) >= 10,
                    "composers_ge_8": len(composers) >= 8, "score_matched_works_ge_8": len(works) >= 8}
    result = {"scope": "File-only PianoVAM natural-P3 preflight; no tracker outcomes",
              "counts": {"acquired": len(rows), "file_eligible": len(good),
                         "eligible_work_families": len(works), "eligible_composers": len(composers),
                         "fail_reasons": dict(Counter(key for row in rows for key,value in row["gates"].items() if not value))},
              "cohort_gates": cohort_gates, "continue_to_segmentation": all(cohort_gates.values()),
              "validation_eligible": False, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
