#!/usr/bin/env python3
"""Outcome-blind monotone clock adapter for public SKY-Piano samples."""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p3_sky_piano_sample_preflight/manifest.json"
OUT = ROOT / "results/p3_sky_piano_clock_adapter"
MIN_RETAINED = 0.95
MIN_DURATION_COVERAGE = 0.90
MAX_GAP_SEC = 6.0
MIN_PILOT_DURATION_SEC = 90.0
MIN_PILOT_ANCHORS = 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_nondecreasing_indices(values):
    """Indices of a deterministic maximum-cardinality nondecreasing subsequence."""
    values = np.asarray(values, dtype=float)
    tails: list[float] = []
    tail_indices: list[int] = []
    previous = np.full(len(values), -1, dtype=int)
    for index, value in enumerate(values):
        slot = bisect_right(tails, float(value))
        if slot:
            previous[index] = tail_indices[slot - 1]
        if slot == len(tails):
            tails.append(float(value))
            tail_indices.append(index)
        else:
            tails[slot] = float(value)
            tail_indices[slot] = index
    if not tail_indices:
        return np.asarray([], dtype=int)
    chain = []
    cursor = tail_indices[-1]
    while cursor >= 0:
        chain.append(cursor)
        cursor = int(previous[cursor])
    return np.asarray(chain[::-1], dtype=int)


def raw_anchors(alignment):
    grouped: dict[float, list[float]] = {}
    for match in alignment["matches"]:
        grouped.setdefault(float(match["score_onset_quarter"]), []).append(
            float(match["midi_time_sec"])
        )
    score = np.asarray(sorted(grouped), dtype=float)
    performance = np.asarray([np.median(grouped[x]) for x in score], dtype=float)
    return score, performance


def adapt_clock(score, performance):
    score = np.asarray(score, dtype=float)
    performance = np.asarray(performance, dtype=float)
    keep = longest_nondecreasing_indices(performance)
    score_kept, performance_kept = score[keep], performance[keep]

    # A quantized MIDI time can correspond to consecutive written positions.
    # Collapse it rather than inventing sub-frame timing.
    unique_time = np.unique(performance_kept)
    score_clock = np.asarray(
        [np.median(score_kept[performance_kept == t]) for t in unique_time], dtype=float
    )
    if len(score_clock) > 1 and np.any(np.diff(score_clock) <= 0):
        raise AssertionError("collapsed score clock is not strict")
    return score_clock, unique_time, keep


def adapt_row(row):
    files = {key: ROOT / value for key, value in row["files"].items()}
    hashes_ok = all(sha256(files[key]) == value for key, value in row["sha256"].items())
    alignment = json.loads(files["alignment.json"].read_text(encoding="utf8"))
    identifiers = [match.get("score_id") for match in alignment["matches"]]
    ids_present_unique = bool(
        identifiers and all(x is not None for x in identifiers) and len(set(identifiers)) == len(identifiers)
    )
    raw_score, raw_performance = raw_anchors(alignment)
    score_clock_quarter, performance_clock_sec, keep = adapt_clock(raw_score, raw_performance)
    retained_fraction = len(keep) / len(raw_score) if len(raw_score) else 0.0
    coverage = (
        (performance_clock_sec[-1] - performance_clock_sec[0]) / row["duration_sec"]
        if len(performance_clock_sec) > 1
        else 0.0
    )
    maximum_gap = float(np.max(np.diff(performance_clock_sec))) if len(performance_clock_sec) > 1 else None
    gates = {
        "hashes_match": hashes_ok,
        "score_ids_present_unique": ids_present_unique,
        "retained_fraction_ge_0_95": bool(retained_fraction >= MIN_RETAINED),
        "duration_coverage_ge_0_90": bool(coverage >= MIN_DURATION_COVERAGE),
        "maximum_gap_le_6_sec": maximum_gap is not None and maximum_gap <= MAX_GAP_SEC,
        "score_clock_strict": bool(len(score_clock_quarter) > 1 and np.all(np.diff(score_clock_quarter) > 0)),
        "performance_clock_strict": bool(len(performance_clock_sec) > 1 and np.all(np.diff(performance_clock_sec) > 0)),
    }
    passed = all(gates.values())
    pilot_eligible = bool(
        passed
        and row["duration_sec"] >= MIN_PILOT_DURATION_SEC
        and len(score_clock_quarter) >= MIN_PILOT_ANCHORS
        and not row["history_exclusion"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    clock_path = OUT / f"{row['trial_id']}_clock.npz"
    np.savez_compressed(
        clock_path,
        score_clock_quarter=score_clock_quarter,
        performance_clock_sec=performance_clock_sec,
        retained_raw_indices=keep,
    )
    return {
        "trial_id": row["trial_id"],
        "duration_sec": row["duration_sec"],
        "history_exclusion": row["history_exclusion"],
        "raw_anchors": len(raw_score),
        "retained_raw_anchors": len(keep),
        "output_anchors": len(score_clock_quarter),
        "omitted_raw_anchors": len(raw_score) - len(keep),
        "retained_fraction": retained_fraction,
        "duration_coverage": coverage,
        "maximum_gap_sec": maximum_gap,
        "gates": gates,
        "adapter_passed": passed,
        "pilot_eligible": pilot_eligible,
        "clock": str(clock_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main():
    source = json.loads(SOURCE.read_text(encoding="utf8"))
    rows = [adapt_row(row) for row in source["rows"]]
    result = {
        "scope": "SKY-Piano public sample alignment clocks; no tracker or temporal outcomes",
        "method": "maximum-cardinality nondecreasing anchor subsequence, then equal-time collapse",
        "fixed_gates": {
            "minimum_retained_fraction": MIN_RETAINED,
            "minimum_duration_coverage": MIN_DURATION_COVERAGE,
            "maximum_anchor_gap_sec": MAX_GAP_SEC,
            "minimum_pilot_duration_sec": MIN_PILOT_DURATION_SEC,
            "minimum_pilot_anchors": MIN_PILOT_ANCHORS,
        },
        "counts": {
            "samples": len(rows),
            "adapter_passed": sum(row["adapter_passed"] for row in rows),
            "pilot_eligible": sum(row["pilot_eligible"] for row in rows),
        },
        "rows": rows,
        "tracker_outcomes_opened": False,
        "temporal_outcomes_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
