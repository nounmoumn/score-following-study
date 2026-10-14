#!/usr/bin/env python3
"""Outcome-blind restart candidates for the frozen natural-P3 cohort."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
ACQUISITION = ROOT / "results/p3_pianovam_acquisition.json"
PREFLIGHT = ROOT / "results/p3_pianovam_file_preflight.json"
MULTISOURCE = ROOT / "results/p3_natural_multisource_preflight.json"
OUT = ROOT / "results/p3_natural_restart_candidates.json"
CSV_OUT = ROOT / "results/p3_natural_restart_review.csv"

NGRAM_NOTES = 5
HARD_PAUSE_SEC = 5.0
MAX_PERF_NOTE_GAP = 75
NEIGHBORHOOD_NOTES = 30
MIN_LOCAL_ANCHORS = 3
MIN_SCORE_DROP_NOTES = 100
MIN_SCORE_DROP_FRACTION = 0.10
MERGE_SEC = 5.0
WEAK_COVERAGE = 0.01


def load_notes(path: Path) -> list[tuple[float, float, int]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    return sorted(
        ((float(n.start), float(n.end), int(n.pitch))
         for instrument in midi.instruments for n in instrument.notes),
        key=lambda value: (value[0], value[2], value[1]),
    )


def unique_score_ngrams(score_pitches: list[int], n: int = NGRAM_NOTES) -> dict[tuple[int, ...], int]:
    locations: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(max(0, len(score_pitches) - n + 1)):
        locations[tuple(score_pitches[index:index + n])].append(index)
    return {gram: values[0] for gram, values in locations.items() if len(values) == 1}


def exact_anchors(performance_pitches: list[int], score_pitches: list[int], n: int = NGRAM_NOTES):
    lookup = unique_score_ngrams(score_pitches, n)
    return [
        {"performance_index": index, "score_index": lookup[gram]}
        for index in range(max(0, len(performance_pitches) - n + 1))
        if (gram := tuple(performance_pitches[index:index + n])) in lookup
    ]


def restart_pairs(anchors: list[dict], score_note_count: int) -> list[dict]:
    """Return conservative backward-jump pairs with local support."""
    required_drop = max(MIN_SCORE_DROP_NOTES, int(np.ceil(MIN_SCORE_DROP_FRACTION * score_note_count)))
    candidates = []
    for left, right in zip(anchors, anchors[1:]):
        if right["performance_index"] - left["performance_index"] > MAX_PERF_NOTE_GAP:
            continue
        if left["score_index"] - right["score_index"] < required_drop:
            continue
        before = [a["score_index"] for a in anchors
                  if left["performance_index"] - NEIGHBORHOOD_NOTES <= a["performance_index"] <= left["performance_index"]]
        after = [a["score_index"] for a in anchors
                 if right["performance_index"] <= a["performance_index"] <= right["performance_index"] + NEIGHBORHOOD_NOTES]
        if len(before) < MIN_LOCAL_ANCHORS or len(after) < MIN_LOCAL_ANCHORS:
            continue
        local_drop = float(np.median(before) - np.median(after))
        if local_drop >= required_drop:
            candidates.append({
                "performance_index": right["performance_index"],
                "score_before_index": left["score_index"],
                "score_after_index": right["score_index"],
                "pair_drop_notes": left["score_index"] - right["score_index"],
                "local_median_drop_notes": local_drop,
            })
    return candidates


def hard_pauses(notes: list[tuple[float, float, int]]) -> list[dict]:
    boundaries = []
    latest_end = notes[0][1] if notes else 0.0
    for index in range(1, len(notes)):
        gap = notes[index][0] - latest_end
        if gap >= HARD_PAUSE_SEC:
            boundaries.append({"performance_index": index, "time_sec": notes[index][0], "gap_sec": gap})
        latest_end = max(latest_end, notes[index][1])
    return boundaries


def merge_candidates(candidates: list[dict], merge_sec: float = MERGE_SEC) -> list[dict]:
    merged = []
    for candidate in sorted(candidates, key=lambda row: row["time_sec"]):
        if merged and candidate["time_sec"] - merged[-1]["time_sec"] <= merge_sec:
            current = merged[-1]
            current["signals"] = sorted(set(current["signals"] + candidate["signals"]))
            current["members"].extend(candidate["members"])
            if "content_restart" in candidate["signals"]:
                current["time_sec"] = candidate["time_sec"]
                current["performance_index"] = candidate["performance_index"]
        else:
            merged.append({**candidate, "members": list(candidate["members"])})
    return merged


def context(notes, index: int, radius: int = 6):
    lo, hi = max(0, index - radius), min(len(notes), index + radius + 1)
    return [{"index": i, "time_sec": notes[i][0], "pitch": notes[i][2]} for i in range(lo, hi)]


def process(record: dict) -> dict:
    performance_path = ROOT / record["performance_path"]
    score_path = ROOT / record["score_midi_path"]
    performance = load_notes(performance_path)
    score = load_notes(score_path)
    pp = [note[2] for note in performance]
    sp = [note[2] for note in score]
    anchors = exact_anchors(pp, sp)
    possible = max(1, len(pp) - NGRAM_NOTES + 1)
    coverage = len(anchors) / possible
    raw = []
    for pause in hard_pauses(performance):
        raw.append({"time_sec": pause["time_sec"], "performance_index": pause["performance_index"],
                    "signals": ["hard_pause"], "members": [pause]})
    for restart in restart_pairs(anchors, len(score)):
        index = restart["performance_index"]
        raw.append({"time_sec": performance[index][0], "performance_index": index,
                    "signals": ["content_restart"], "members": [restart]})
    merged = merge_candidates(raw)
    for number, candidate in enumerate(merged, start=1):
        candidate["candidate_id"] = f"{record['record_time']}-B{number:03d}"
        candidate["context"] = context(performance, candidate["performance_index"])
    performance_duration = max(note[1] for note in performance)
    score_duration = max(note[1] for note in score)
    duration_ratio = float(performance_duration / score_duration)
    whole_take_plausible = 0.75 <= duration_ratio <= 1.35
    return {
        "record_time": record["record_time"], "composer": record["score_composer"],
        "piece": record["piece"], "work_id": str(record["rule_index"]),
        "performance_path": record["performance_path"], "score_midi_path": record["score_midi_path"],
        "performance_notes": len(performance), "score_notes": len(score),
        "duration_ratio": duration_ratio, "unique_anchor_count": len(anchors),
        "unique_anchor_coverage": coverage, "content_mapping_weak": coverage < WEAK_COVERAGE,
        "whole_take_plausible": whole_take_plausible, "candidate_count": len(merged),
        "review_ready": whole_take_plausible or bool(merged), "candidates": merged,
    }


def records():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    eligible = {row["record_time"] for row in preflight["rows"] if row["file_eligible"]}
    rows = [row for row in acquisition["rows"] if row["record_time"] in eligible]
    supplement = json.loads(MULTISOURCE.read_text(encoding="utf-8"))["supplement"]
    if supplement["eligible"]:
        rows.append({
            "record_time": "SMD_Brahms_Op010No1_003_20090916", "score_composer": "Brahms,_Johannes",
            "piece": "Piano Piece Op.10 No.1", "rule_index": "SMD_BRAHMS_OP10_1",
            "performance_path": "data/pianovam_p3_natural/supplement_smd_brahms/performance.mid",
            "score_midi_path": "data/pianovam_p3_natural/supplement_smd_brahms/score.mid",
            "performance_duration_sec": supplement["performance_duration_sec"],
            "score_duration_sec": supplement["score_duration_sec"],
        })
    return rows


def main():
    rows = [process(record) for record in records()]
    signals = Counter(signal for row in rows for candidate in row["candidates"] for signal in candidate["signals"])
    gates = {
        "processed_all_25": len(rows) == 25,
        "candidate_provenance_complete": all(
            candidate.get("signals") and candidate.get("context")
            for row in rows for candidate in row["candidates"]),
        "review_ready_recordings_ge_20": sum(row["review_ready"] for row in rows) >= 20,
        "strong_content_mapping_ge_20": sum(not row["content_mapping_weak"] for row in rows) >= 20,
    }
    result = {
        "scope": "Outcome-blind restart candidate generation; no score-follower or P3 outcomes",
        "parameters": {"ngram_notes": NGRAM_NOTES, "hard_pause_sec": HARD_PAUSE_SEC,
                       "max_performance_note_gap": MAX_PERF_NOTE_GAP,
                       "neighborhood_notes": NEIGHBORHOOD_NOTES, "minimum_local_anchors": MIN_LOCAL_ANCHORS,
                       "minimum_score_drop_notes": MIN_SCORE_DROP_NOTES,
                       "minimum_score_drop_fraction": MIN_SCORE_DROP_FRACTION,
                       "merge_sec": MERGE_SEC, "weak_coverage": WEAK_COVERAGE},
        "counts": {"recordings": len(rows), "candidates": sum(row["candidate_count"] for row in rows),
                   "candidate_signals": dict(signals),
                   "whole_take_plausible": sum(row["whole_take_plausible"] for row in rows),
                   "review_ready": sum(row["review_ready"] for row in rows),
                   "content_mapping_weak": sum(row["content_mapping_weak"] for row in rows)},
        "gates": gates, "continue_to_manual_review": all(gates.values()),
        "validation_eligible": False, "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["candidate_id", "record_time", "composer", "piece", "time_sec", "signals",
                  "performance_index", "context_pitches", "review_decision", "reviewer", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            for candidate in row["candidates"]:
                writer.writerow({"candidate_id": candidate["candidate_id"], "record_time": row["record_time"],
                    "composer": row["composer"], "piece": row["piece"], "time_sec": candidate["time_sec"],
                    "signals": "+".join(candidate["signals"]), "performance_index": candidate["performance_index"],
                    "context_pitches": " ".join(str(item["pitch"]) for item in candidate["context"]),
                    "review_decision": "", "reviewer": "", "review_reason": ""})
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
