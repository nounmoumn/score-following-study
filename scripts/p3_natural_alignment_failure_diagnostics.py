#!/usr/bin/env python3
"""Describe failed pilot intervals without reading tracker or P3 outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scripts import p3_natural_alignment_calibration as calibration
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_alignment_calibration as calibration
    import p3_natural_event_restart_candidates as event

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_natural_alignment_failure_diagnostics.json"


def unmapped_runs(mapping, events):
    runs = []
    start = None
    for index in range(len(mapping) + 1):
        unmapped = index < len(mapping) and mapping[index] < 0
        if unmapped and start is None:
            start = index
        if not unmapped and start is not None:
            stop = index
            before = int(mapping[start - 1]) if start else None
            after = int(mapping[stop]) if stop < len(mapping) else None
            end_time = events[stop]["time_sec"] if stop < len(events) else events[-1]["latest_end_sec"]
            runs.append({
                "start_event": start,
                "stop_event_exclusive": stop,
                "event_count": stop - start,
                "start_sec": float(events[start]["time_sec"]),
                "end_sec": float(end_time),
                "duration_sec": float(end_time - events[start]["time_sec"]),
                "mapped_score_before": before,
                "mapped_score_after": after,
                "score_jump_across": None if before is None or after is None else after - before,
                "median_pitch_count": float(np.median([len(events[i]["pitches"]) for i in range(start, stop)])),
            })
            start = None
    return sorted(runs, key=lambda row: (-row["duration_sec"], -row["event_count"]))


def main():
    rows = []
    for record in calibration.pilot_records():
        score = event.onset_events(event.load_notes(ROOT / record["score_midi_path"]), event.SCORE_ONSET_TOLERANCE)
        performance = event.onset_events(event.load_notes(ROOT / record["performance_path"]), event.PERFORMANCE_ONSET_TOLERANCE)
        exact = calibration.align(score, performance, False)
        chroma = calibration.align(score, performance, True)
        exact_runs = unmapped_runs(exact, performance)
        chroma_runs = unmapped_runs(chroma, performance)
        rows.append({
            "record_time": record["record_time"],
            "score_events": len(score),
            "performance_events": len(performance),
            "exact_unmapped_events": int(np.sum(exact < 0)),
            "chroma_unmapped_events": int(np.sum(chroma < 0)),
            "exact_unmapped_runs": len(exact_runs),
            "chroma_unmapped_runs": len(chroma_runs),
            "exact_single_event_run_share": float(np.mean([r["event_count"] == 1 for r in exact_runs])) if exact_runs else 0.0,
            "chroma_single_event_run_share": float(np.mean([r["event_count"] == 1 for r in chroma_runs])) if chroma_runs else 0.0,
            "exact_top_runs": exact_runs[:10],
            "chroma_top_runs": chroma_runs[:10],
        })
    result = {
        "scope": "Outcome-blind failure localization; no tracker/P3 outcomes",
        "interpretation_limit": "Descriptive only; does not revise the frozen calibration gate.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
