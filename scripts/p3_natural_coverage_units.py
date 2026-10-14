#!/usr/bin/env python3
"""Build provisional score-coverage units without running the P3 tracker."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from scripts import p3_natural_event_restart_candidates as event_method
except ImportError:
    import p3_natural_event_restart_candidates as event_method

ROOT = Path(__file__).resolve().parents[1]
METHOD_B = ROOT / "results/p3_natural_event_restart_candidates.json"
OUT = ROOT / "results/p3_natural_coverage_units.json"
CSV_OUT = ROOT / "results/p3_natural_coverage_unit_review.csv"

MIN_DURATION_SEC = 30.0
MIN_ANCHORS = 20
MIN_SCORE_SPAN = 0.25
MIN_MONOTONE_SHARE = 0.90
BACKWARD_TOLERANCE = 0.02


def unit_metrics(anchors, events, score_event_count, start_index, end_index):
    selected = [a for a in anchors if start_index <= a["performance_index"] < end_index]
    start_time = events[start_index]["time_sec"] if start_index < len(events) else events[-1]["time_sec"]
    end_time = events[end_index]["time_sec"] if end_index < len(events) else events[-1]["latest_end_sec"]
    if selected:
        normalized = np.array([a["score_index"] for a in selected], dtype=float) / max(1, score_event_count - 1)
        q05, q95 = np.percentile(normalized, [5, 95])
        monotone = float(np.mean(np.diff(normalized) >= -BACKWARD_TOLERANCE)) if len(normalized) > 1 else 0.0
        edge = max(1, len(normalized) // 5)
        net_forward = bool(np.median(normalized[-edge:]) >= np.median(normalized[:edge]))
    else:
        q05 = q95 = monotone = 0.0
        net_forward = False
    span = float(q95 - q05)
    gates = {"duration_ge_30": end_time-start_time >= MIN_DURATION_SEC,
             "anchors_ge_20": len(selected) >= MIN_ANCHORS,
             "score_span_ge_0_25": span >= MIN_SCORE_SPAN,
             "monotone_share_ge_0_90": monotone >= MIN_MONOTONE_SHARE,
             "net_direction_forward": net_forward}
    return {"start_event_index": start_index, "end_event_index_exclusive": end_index,
            "start_sec": start_time, "end_sec": end_time, "duration_sec": end_time-start_time,
            "anchor_count": len(selected), "score_position_q05": float(q05),
            "score_position_q95": float(q95), "score_span": span,
            "monotone_share": monotone, "net_direction_forward": net_forward,
            "gates": gates, "reviewable": all(gates.values())}


def process(record, method_b_row):
    performance = event_method.load_notes(ROOT / record["performance_path"])
    score = event_method.load_notes(ROOT / record["score_midi_path"])
    pe = event_method.onset_events(performance, event_method.PERFORMANCE_ONSET_TOLERANCE)
    se = event_method.onset_events(score, event_method.SCORE_ONSET_TOLERANCE)
    anchors = event_method.event_anchors([e["token"] for e in pe], [e["token"] for e in se])
    cuts = sorted(set([0, len(pe)] + [int(c["performance_index"]) for c in method_b_row["candidates"]]))
    units = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), 1):
        if end <= start:
            continue
        unit = unit_metrics(anchors, pe, len(se), start, end)
        unit["unit_id"] = f"{record['record_time']}-U{index:03d}"
        unit["left_boundary"] = "file_start" if start == 0 else "candidate"
        unit["right_boundary"] = "file_end" if end == len(pe) else "candidate"
        units.append(unit)
    return {"record_time": record["record_time"], "composer": record["score_composer"],
            "piece": record["piece"], "candidate_boundaries": len(method_b_row["candidates"]),
            "unit_count": len(units), "reviewable_units": sum(u["reviewable"] for u in units), "units": units}


def main():
    method_b = json.loads(METHOD_B.read_text(encoding="utf-8"))
    by_id = {row["record_time"]: row for row in method_b["rows"]}
    rows = [process(record, by_id[record["record_time"]]) for record in event_method.records()]
    ready = [row for row in rows if row["reviewable_units"]]
    gates = {"recordings_with_reviewable_unit_ge_20": len(ready) >= 20,
             "represented_composers_ge_8": len({row["composer"] for row in ready}) >= 8,
             "all_units_have_gate_provenance": all("gates" in unit for row in rows for unit in row["units"])}
    result = {"scope": "Outcome-blind provisional score-coverage units; no tracker/P3 outcomes",
        "parameters": {"minimum_duration_sec": MIN_DURATION_SEC, "minimum_anchors": MIN_ANCHORS,
            "minimum_score_span": MIN_SCORE_SPAN, "minimum_monotone_share": MIN_MONOTONE_SHARE,
            "backward_tolerance_score_fraction": BACKWARD_TOLERANCE},
        "counts": {"recordings": len(rows), "recordings_with_reviewable_unit": len(ready),
            "reviewable_units": sum(row["reviewable_units"] for row in rows),
            "represented_composers": len({row["composer"] for row in ready})},
        "gates": gates, "continue_to_boundary_review": all(gates.values()),
        "validation_eligible": False, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["unit_id", "record_time", "composer", "piece", "start_sec", "end_sec", "anchor_count",
                  "score_span", "monotone_share", "reviewable", "review_decision", "reviewer", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            for unit in row["units"]:
                writer.writerow({**{key: unit[key] for key in ("unit_id", "start_sec", "end_sec", "anchor_count",
                    "score_span", "monotone_share", "reviewable")}, "record_time": row["record_time"],
                    "composer": row["composer"], "piece": row["piece"], "review_decision": "",
                    "reviewer": "", "review_reason": ""})
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
