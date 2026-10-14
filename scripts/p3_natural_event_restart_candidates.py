#!/usr/bin/env python3
"""Method-B onset-event restart candidates, independent of P3 outcomes."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.p3_natural_restart_candidates import hard_pauses, load_notes, merge_candidates, records
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from p3_natural_restart_candidates import hard_pauses, load_notes, merge_candidates, records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_natural_event_restart_candidates.json"
CSV_OUT = ROOT / "results/p3_natural_event_restart_review.csv"

NGRAM_EVENTS = 4
PERFORMANCE_ONSET_TOLERANCE = 0.045
SCORE_ONSET_TOLERANCE = 0.001
MAX_PERFORMANCE_EVENT_GAP = 40
NEIGHBORHOOD_EVENTS = 15
MIN_LOCAL_ANCHORS = 3
MIN_SCORE_DROP_EVENTS = 25
MIN_SCORE_DROP_FRACTION = 0.10
WEAK_COVERAGE = 0.01


def onset_events(notes, tolerance: float):
    groups = []
    for start, end, pitch in notes:
        if not groups or start - groups[-1]["time_sec"] > tolerance:
            groups.append({"time_sec": start, "latest_end_sec": end, "pitches": [pitch]})
        else:
            groups[-1]["latest_end_sec"] = max(groups[-1]["latest_end_sec"], end)
            groups[-1]["pitches"].append(pitch)
    for group in groups:
        group["token"] = max(group["pitches"])
    return groups


def unique_event_ngrams(tokens, n=NGRAM_EVENTS):
    locations = defaultdict(list)
    for index in range(max(0, len(tokens) - n + 1)):
        locations[tuple(tokens[index:index + n])].append(index)
    return {gram: indices[0] for gram, indices in locations.items() if len(indices) == 1}


def event_anchors(performance_tokens, score_tokens, n=NGRAM_EVENTS):
    lookup = unique_event_ngrams(score_tokens, n)
    return [{"performance_index": index, "score_index": lookup[gram]}
            for index in range(max(0, len(performance_tokens) - n + 1))
            if (gram := tuple(performance_tokens[index:index + n])) in lookup]


def event_restart_pairs(anchors, score_event_count):
    required = max(MIN_SCORE_DROP_EVENTS, int(np.ceil(MIN_SCORE_DROP_FRACTION * score_event_count)))
    output = []
    for left, right in zip(anchors, anchors[1:]):
        if right["performance_index"] - left["performance_index"] > MAX_PERFORMANCE_EVENT_GAP:
            continue
        if left["score_index"] - right["score_index"] < required:
            continue
        before = [a["score_index"] for a in anchors
                  if left["performance_index"] - NEIGHBORHOOD_EVENTS <= a["performance_index"] <= left["performance_index"]]
        after = [a["score_index"] for a in anchors
                 if right["performance_index"] <= a["performance_index"] <= right["performance_index"] + NEIGHBORHOOD_EVENTS]
        if len(before) >= MIN_LOCAL_ANCHORS and len(after) >= MIN_LOCAL_ANCHORS:
            local_drop = float(np.median(before) - np.median(after))
            if local_drop >= required:
                output.append({"performance_event_index": right["performance_index"],
                    "score_before_event": left["score_index"], "score_after_event": right["score_index"],
                    "pair_drop_events": left["score_index"] - right["score_index"],
                    "local_median_drop_events": local_drop})
    return output


def event_context(events, index, radius=5):
    return [{"index": i, "time_sec": events[i]["time_sec"], "top_pitch": events[i]["token"]}
            for i in range(max(0, index-radius), min(len(events), index+radius+1))]


def process(record):
    performance = load_notes(ROOT / record["performance_path"])
    score = load_notes(ROOT / record["score_midi_path"])
    pe = onset_events(performance, PERFORMANCE_ONSET_TOLERANCE)
    se = onset_events(score, SCORE_ONSET_TOLERANCE)
    anchors = event_anchors([e["token"] for e in pe], [e["token"] for e in se])
    coverage = len(anchors) / max(1, len(pe) - NGRAM_EVENTS + 1)
    raw = []
    for pause in hard_pauses(performance):
        event_index = int(np.searchsorted([e["time_sec"] for e in pe], pause["time_sec"]))
        event_index = min(event_index, len(pe)-1)
        raw.append({"time_sec": pause["time_sec"], "performance_index": event_index,
                    "signals": ["hard_pause"], "members": [pause]})
    for restart in event_restart_pairs(anchors, len(se)):
        index = restart["performance_event_index"]
        raw.append({"time_sec": pe[index]["time_sec"], "performance_index": index,
                    "signals": ["event_content_restart"], "members": [restart]})
    candidates = merge_candidates(raw)
    for number, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"{record['record_time']}-E{number:03d}"
        candidate["context"] = event_context(pe, candidate["performance_index"])
    ratio = max(n[1] for n in performance) / max(n[1] for n in score)
    whole = 0.75 <= ratio <= 1.35
    return {"record_time": record["record_time"], "composer": record["score_composer"],
        "piece": record["piece"], "work_id": str(record["rule_index"]),
        "performance_path": record["performance_path"], "score_midi_path": record["score_midi_path"],
        "performance_events": len(pe), "score_events": len(se), "duration_ratio": ratio,
        "unique_anchor_count": len(anchors), "unique_anchor_coverage": coverage,
        "content_mapping_weak": coverage < WEAK_COVERAGE, "whole_take_plausible": whole,
        "candidate_count": len(candidates), "review_ready": whole or bool(candidates), "candidates": candidates}


def main():
    rows = [process(record) for record in records()]
    signals = Counter(s for row in rows for c in row["candidates"] for s in c["signals"])
    gates = {"processed_all_25": len(rows) == 25,
        "candidate_provenance_complete": all(c.get("signals") and c.get("context") for row in rows for c in row["candidates"]),
        "strong_content_mapping_ge_20": sum(not row["content_mapping_weak"] for row in rows) >= 20,
        "review_ready_recordings_ge_20": sum(row["review_ready"] for row in rows) >= 20}
    result = {"scope": "Outcome-blind onset-event restart candidates (method B); no P3 outcomes",
        "parameters": {"ngram_events": NGRAM_EVENTS, "performance_onset_tolerance_sec": PERFORMANCE_ONSET_TOLERANCE,
            "score_onset_tolerance_sec": SCORE_ONSET_TOLERANCE, "max_performance_event_gap": MAX_PERFORMANCE_EVENT_GAP,
            "neighborhood_events": NEIGHBORHOOD_EVENTS, "minimum_local_anchors": MIN_LOCAL_ANCHORS,
            "minimum_score_drop_events": MIN_SCORE_DROP_EVENTS, "minimum_score_drop_fraction": MIN_SCORE_DROP_FRACTION,
            "weak_coverage": WEAK_COVERAGE},
        "counts": {"recordings": len(rows), "candidates": sum(r["candidate_count"] for r in rows),
            "candidate_signals": dict(signals), "whole_take_plausible": sum(r["whole_take_plausible"] for r in rows),
            "review_ready": sum(r["review_ready"] for r in rows),
            "content_mapping_weak": sum(r["content_mapping_weak"] for r in rows)},
        "gates": gates, "continue_to_manual_review": all(gates.values()), "validation_eligible": False, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["candidate_id", "record_time", "composer", "piece", "time_sec", "signals",
                  "performance_event_index", "context_top_pitches", "review_decision", "reviewer", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            for candidate in row["candidates"]:
                writer.writerow({"candidate_id": candidate["candidate_id"], "record_time": row["record_time"],
                    "composer": row["composer"], "piece": row["piece"], "time_sec": candidate["time_sec"],
                    "signals": "+".join(candidate["signals"]), "performance_event_index": candidate["performance_index"],
                    "context_top_pitches": " ".join(str(x["top_pitch"]) for x in candidate["context"]),
                    "review_decision": "", "reviewer": "", "review_reason": ""})
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
