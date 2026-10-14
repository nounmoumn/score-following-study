#!/usr/bin/env python3
"""Evaluate the frozen many-to-many path clock on annotated ASAP works."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

try:
    from scripts import p3_natural_alignment_calibration as base
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_alignment_calibration as base
    import p3_natural_event_restart_candidates as event

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_natural_path_clock_manifest.json"
OUT = ROOT / "results/p3_natural_path_clock_calibration.json"
GAP_SURCHARGE = 0.2


@njit(cache=False)
def path_clock_masks(score_lo, score_hi, perf_lo, perf_hi, gap_surcharge=GAP_SURCHARGE):
    n, m = len(score_lo), len(perf_lo)
    trace = np.zeros((n, m), dtype=np.uint8)
    previous = np.full(m, np.inf, dtype=np.float64)
    current = np.full(m, np.inf, dtype=np.float64)
    previous[0] = base.jaccard_distance(score_lo[0], score_hi[0], perf_lo[0], perf_hi[0])
    for j in range(1, m):
        local = base.jaccard_distance(score_lo[0], score_hi[0], perf_lo[j], perf_hi[j])
        previous[j] = previous[j - 1] + gap_surcharge + local
        trace[0, j] = 2
    for i in range(1, n):
        local = base.jaccard_distance(score_lo[i], score_hi[i], perf_lo[0], perf_hi[0])
        current[0] = previous[0] + gap_surcharge + local
        trace[i, 0] = 1
        for j in range(1, m):
            local = base.jaccard_distance(score_lo[i], score_hi[i], perf_lo[j], perf_hi[j])
            diagonal = previous[j - 1]
            up = previous[j] + gap_surcharge
            left = current[j - 1] + gap_surcharge
            if diagonal <= up and diagonal <= left:
                current[j] = diagonal + local
                trace[i, j] = 0
            elif up <= left:
                current[j] = up + local
                trace[i, j] = 1
            else:
                current[j] = left + local
                trace[i, j] = 2
        previous, current = current, previous
    minimum = np.full(m, n, dtype=np.int64)
    maximum = np.full(m, -1, dtype=np.int64)
    i, j = n - 1, m - 1
    while True:
        if i < minimum[j]:
            minimum[j] = i
        if i > maximum[j]:
            maximum[j] = i
        if i == 0 and j == 0:
            break
        direction = trace[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return (minimum + maximum) // 2


def path_clock(score_events, performance_events, chroma=False):
    score_lo, score_hi = base.masks(score_events, chroma)
    perf_lo, perf_hi = base.masks(performance_events, chroma)
    return path_clock_masks(score_lo, score_hi, perf_lo, perf_hi)


def maximum_plateau_seconds(mapping, performance_events):
    maximum = 0.0
    start = 0
    for stop in range(1, len(mapping) + 1):
        if stop == len(mapping) or mapping[stop] != mapping[start]:
            maximum = max(maximum, performance_events[stop - 1]["time_sec"] - performance_events[start]["time_sec"])
            start = stop
    return float(maximum)


def load_times(path):
    values = np.loadtxt(path, usecols=0, ndmin=1)
    return np.asarray(values, dtype=float)


def summarize(row):
    score = event.onset_events(event.load_notes(ROOT / row["score_midi"]), event.SCORE_ONSET_TOLERANCE)
    performance = event.onset_events(event.load_notes(ROOT / row["performance_midi"]), event.PERFORMANCE_ONSET_TOLERANCE)
    exact = path_clock(score, performance, False)
    chroma = path_clock(score, performance, True)
    perf_annotation = load_times(ROOT / row["performance_annotations"])
    score_annotation = load_times(ROOT / row["score_annotations"])
    if len(perf_annotation) != len(score_annotation):
        raise ValueError(f"annotation length mismatch: {row['work']}")
    perf_event_times = np.asarray([group["time_sec"] for group in performance])
    score_event_times = np.asarray([group["time_sec"] for group in score])
    support = (perf_annotation >= perf_event_times[0]) & (perf_annotation <= perf_event_times[-1])
    predicted_exact = np.interp(perf_annotation[support], perf_event_times, score_event_times[exact])
    predicted_chroma = np.interp(perf_annotation[support], perf_event_times, score_event_times[chroma])
    residual = np.abs(predicted_exact - score_annotation[support])
    disagreement = np.abs(predicted_exact - predicted_chroma)
    exact_plateau = maximum_plateau_seconds(exact, performance)
    chroma_plateau = maximum_plateau_seconds(chroma, performance)
    infrastructure = {
        "exact_all_mapped": bool(np.all(exact >= 0)),
        "chroma_all_mapped": bool(np.all(chroma >= 0)),
        "exact_monotone": bool(np.all(np.diff(exact) >= 0)),
        "chroma_monotone": bool(np.all(np.diff(chroma) >= 0)),
        "exact_plateau_le_5sec": exact_plateau <= 5.0,
        "chroma_plateau_le_5sec": chroma_plateau <= 5.0,
    }
    return {
        "composer": row["composer"],
        "work": row["work"],
        "performance_midi": row["performance_midi"],
        "score_events": len(score),
        "performance_events": len(performance),
        "annotated_beats": len(perf_annotation),
        "evaluated_beats": int(np.sum(support)),
        "outside_support": int(np.sum(~support)),
        "exact_median_abs_score_sec": float(np.median(residual)),
        "exact_p95_abs_score_sec": float(np.percentile(residual, 95)),
        "exact_max_abs_score_sec": float(np.max(residual)),
        "exact_beats_le_0_50": int(np.sum(residual <= 0.50)),
        "method_p95_disagreement_score_sec": float(np.percentile(disagreement, 95)),
        "exact_max_plateau_sec": exact_plateau,
        "chroma_max_plateau_sec": chroma_plateau,
        "infrastructure": infrastructure,
        "residuals": residual.tolist(),
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    rows = []
    for index, row in enumerate(manifest["rows"], 1):
        summary = summarize(row)
        rows.append(summary)
        print(f"[{index:02d}/{len(manifest['rows'])}] {row['work']}: median={summary['exact_median_abs_score_sec']:.4f}s p95={summary['exact_p95_abs_score_sec']:.4f}s", flush=True)
    n = len(rows)
    pooled = np.concatenate([np.asarray(row["residuals"]) for row in rows])
    gates = {
        "works_eq_24": n == 24,
        "composers_ge_6": len({row["composer"] for row in rows}) >= 6,
        "median_pass_share_ge_0_90": sum(row["exact_median_abs_score_sec"] <= 0.05 for row in rows) / n >= 0.90,
        "p95_pass_share_ge_0_90": sum(row["exact_p95_abs_score_sec"] <= 0.25 for row in rows) / n >= 0.90,
        "pooled_beats_le_0_50_share_ge_0_95": float(np.mean(pooled <= 0.50)) >= 0.95,
        "method_agreement_pass_share_ge_0_90": sum(row["method_p95_disagreement_score_sec"] <= 0.25 for row in rows) / n >= 0.90,
        "all_infrastructure_gates": all(all(row["infrastructure"].values()) for row in rows),
    }
    result = {
        "scope": "Frozen path-clock calibration on independent ASAP beat annotations; no tracker/P3 outcomes",
        "parameters": {"score_onset_tolerance_sec": event.SCORE_ONSET_TOLERANCE, "performance_onset_tolerance_sec": event.PERFORMANCE_ONSET_TOLERANCE, "gap_surcharge": GAP_SURCHARGE},
        "summary": {
            "works": n,
            "composers": len({row["composer"] for row in rows}),
            "beats": len(pooled),
            "median_pass_share": sum(row["exact_median_abs_score_sec"] <= 0.05 for row in rows) / n,
            "p95_pass_share": sum(row["exact_p95_abs_score_sec"] <= 0.25 for row in rows) / n,
            "pooled_beats_le_0_50_share": float(np.mean(pooled <= 0.50)),
            "method_agreement_pass_share": sum(row["method_p95_disagreement_score_sec"] <= 0.25 for row in rows) / n,
            "gates": gates,
            "pass": all(gates.values()),
        },
        "continue_after_certification": all(gates.values()),
        "alignment_is_natural_p3_ground_truth": False,
        "validation_eligible": False,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
