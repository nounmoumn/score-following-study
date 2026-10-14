#!/usr/bin/env python3
"""Outcome-blind monotone mapping from unfolded MusicXML to score MIDI."""

import json
import re
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import partitura as pt
import partitura.score as ps

ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = ROOT / "results/p3_r_repeat_structure_audit.json"
RESULT = ROOT / "results/p3_r_repeat_mapping_audit.json"
sys.setrecursionlimit(100000)


def occurrence(note_id):
    match = re.search(r"-(\d+)$", str(note_id))
    return int(match.group(1)) if match else 1


def build_mapping(row):
    path = ROOT / row["path"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml_score = pt.load_score(str(path))
        unfolded = ps.unfold_part_maximal(xml_score, update_ids=True, ignore_leaps=False).note_array()
        midi_perf = pt.load_performance_midi(str(path.parent / "score.mid"), merge_tracks=True, quiet=True).note_array()
    xo = np.lexsort((unfolded["pitch"], unfolded["onset_quarter"]))
    mo = np.lexsort((midi_perf["pitch"], midi_perf["onset_tick"]))
    xp = [int(value) for value in unfolded["pitch"][xo]]
    mp = [int(value) for value in midi_perf["pitch"][mo]]
    matcher = SequenceMatcher(None, xp, mp, autojunk=False)
    pairs = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            xi, mi = xo[block.a + offset], mo[block.b + offset]
            pairs.append((float(midi_perf["onset_sec"][mi]), occurrence(unfolded["id"][xi])))
    matched = len(pairs)
    f1 = 2 * matched / (len(xp) + len(mp)) if xp or mp else 0.0
    # At a chord onset use the greatest occurrence; the new pass starts at its first notes.
    by_time = {}
    for time_sec, occ in pairs:
        by_time[time_sec] = max(occ, by_time.get(time_sec, 0))
    times = np.asarray(sorted(by_time), dtype=float)
    occurrences = np.asarray([by_time[value] for value in times], dtype=int)
    entries = [float(times[i]) for i in range(1, len(times)) if occurrences[i-1] == 1 and occurrences[i] == 2]
    return {"xml_notes": len(xp), "midi_notes": len(mp), "matched_notes": matched,
            "mapping_f1": float(f1), "pass2_entries_score_sec": entries}, times, occurrences


def main():
    source = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    rows = []
    for row in source["rows"]:
        if not row.get("structure_eligible"):
            continue
        metrics, _, _ = build_mapping(row)
        eligible = metrics["mapping_f1"] >= 0.90 and bool(metrics["pass2_entries_score_sec"])
        rows.append({"id": row["id"], "composer": row["composer"], "score_dataset": row["score_dataset"],
                     **metrics, "mapping_eligible": eligible})
        print(row["id"], f"F1={metrics['mapping_f1']:.3f}", f"entries={len(metrics['pass2_entries_score_sec'])}")
    good = [row for row in rows if row["mapping_eligible"]]
    gates = {"eligible_files_ge_8": len(good) >= 8,
             "eligible_composers_ge_5": len({row["composer"] for row in good}) >= 5,
             "eligible_score_datasets_ge_2": len({row["score_dataset"] for row in good}) >= 2}
    result = {"scope": "Outcome-blind unfolded-XML to score-MIDI mapping audit",
              "preserved_validation_requested": 0,
              "summary": {"structure_files": len(rows), "mapping_eligible": len(good),
                          "eligible_composers": len({row["composer"] for row in good}),
                          "eligible_score_datasets": sorted({row["score_dataset"] for row in good})},
              "gates": gates, "continue_to_intervention": all(gates.values()), "rows": rows}
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
