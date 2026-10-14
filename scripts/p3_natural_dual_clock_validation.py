#!/usr/bin/env python3
"""One-shot validation of the frozen dual path-clock selector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scripts import p3_natural_event_restart_candidates as event
    from scripts import p3_natural_path_clock_calibration as clock
except ImportError:
    import p3_natural_event_restart_candidates as event
    import p3_natural_path_clock_calibration as clock

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_natural_dual_clock_manifest.json"
OUT = ROOT / "results/p3_natural_dual_clock_validation.json"


def usable(mapping, performance):
    return bool(
        np.all(mapping >= 0)
        and np.all(np.diff(mapping) >= 0)
        and clock.maximum_plateau_seconds(mapping, performance) <= 5.0
    )


def select_clock(exact_mapping, chroma_mapping, score_times, performance):
    exact_times = score_times[exact_mapping]
    chroma_times = score_times[chroma_mapping]
    exact_usable = usable(exact_mapping, performance)
    chroma_usable = usable(chroma_mapping, performance)
    disagreement = float(np.percentile(np.abs(exact_times - chroma_times), 95))
    if exact_usable and chroma_usable and disagreement <= 0.25:
        return (exact_times + chroma_times) / 2.0, "mean_exact_chroma", disagreement
    if exact_usable and not chroma_usable:
        return exact_times, "exact_only", disagreement
    if chroma_usable and not exact_usable:
        return chroma_times, "chroma_only", disagreement
    return None, "ineligible", disagreement


def evaluate(row):
    score = event.onset_events(event.load_notes(ROOT / row["score_midi"]), event.SCORE_ONSET_TOLERANCE)
    performance = event.onset_events(event.load_notes(ROOT / row["performance_midi"]), event.PERFORMANCE_ONSET_TOLERANCE)
    exact = clock.path_clock(score, performance, False)
    chroma = clock.path_clock(score, performance, True)
    score_times = np.asarray([group["time_sec"] for group in score])
    performance_times = np.asarray([group["time_sec"] for group in performance])
    selected, branch, disagreement = select_clock(exact, chroma, score_times, performance)
    base = {
        "composer": row["composer"],
        "work": row["work"],
        "performance_midi": row["performance_midi"],
        "score_events": len(score),
        "performance_events": len(performance),
        "exact_usable": usable(exact, performance),
        "chroma_usable": usable(chroma, performance),
        "exact_max_plateau_sec": clock.maximum_plateau_seconds(exact, performance),
        "chroma_max_plateau_sec": clock.maximum_plateau_seconds(chroma, performance),
        "method_p95_disagreement_score_sec": disagreement,
        "selector_branch": branch,
        "eligible": selected is not None,
    }
    if selected is None:
        return {**base, "residuals": []}
    perf_annotation = clock.load_times(ROOT / row["performance_annotations"])
    score_annotation = clock.load_times(ROOT / row["score_annotations"])
    if len(perf_annotation) != len(score_annotation):
        raise ValueError(f"annotation length mismatch: {row['work']}")
    support = (perf_annotation >= performance_times[0]) & (perf_annotation <= performance_times[-1])
    prediction = np.interp(perf_annotation[support], performance_times, selected)
    residual = np.abs(prediction - score_annotation[support])
    selected_monotone = bool(np.all(np.diff(selected) >= 0))
    return {
        **base,
        "annotated_beats": len(perf_annotation),
        "evaluated_beats": int(np.sum(support)),
        "outside_support": int(np.sum(~support)),
        "selected_monotone": selected_monotone,
        "median_abs_score_sec": float(np.median(residual)),
        "p95_abs_score_sec": float(np.percentile(residual, 95)),
        "max_abs_score_sec": float(np.max(residual)),
        "beats_le_0_50": int(np.sum(residual <= 0.50)),
        "residuals": residual.tolist(),
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    rows = []
    for index, row in enumerate(manifest["rows"], 1):
        result = evaluate(row)
        rows.append(result)
        metric = "ineligible" if not result["eligible"] else f"median={result['median_abs_score_sec']:.4f}s p95={result['p95_abs_score_sec']:.4f}s"
        print(f"[{index:02d}/{len(manifest['rows'])}] {row['work']}: {result['selector_branch']} {metric}", flush=True)
    eligible = [row for row in rows if row["eligible"]]
    pooled = np.concatenate([np.asarray(row["residuals"]) for row in eligible]) if eligible else np.asarray([])
    eligible_share = len(eligible) / len(rows)
    median_share = sum(row["median_abs_score_sec"] <= 0.05 for row in eligible) / len(eligible) if eligible else 0.0
    p95_share = sum(row["p95_abs_score_sec"] <= 0.25 for row in eligible) / len(eligible) if eligible else 0.0
    pooled_share = float(np.mean(pooled <= 0.50)) if len(pooled) else 0.0
    gates = {
        "works_eq_24": len(rows) == 24,
        "composers_ge_6": len({row["composer"] for row in rows}) >= 6,
        "eligible_share_ge_0_90": eligible_share >= 0.90,
        "median_pass_share_ge_0_90": median_share >= 0.90,
        "p95_pass_share_ge_0_90": p95_share >= 0.90,
        "pooled_beats_le_0_50_share_ge_0_95": pooled_share >= 0.95,
        "all_selected_clocks_monotone": all(row.get("selected_monotone", False) for row in eligible),
    }
    summary = {
        "works": len(rows),
        "composers": len({row["composer"] for row in rows}),
        "eligible_works": len(eligible),
        "eligible_share": eligible_share,
        "evaluated_beats": len(pooled),
        "median_pass_share": median_share,
        "p95_pass_share": p95_share,
        "pooled_beats_le_0_50_share": pooled_share,
        "selector_branches": {branch: sum(row["selector_branch"] == branch for row in rows) for branch in ("mean_exact_chroma", "exact_only", "chroma_only", "ineligible")},
        "gates": gates,
        "pass": all(gates.values()),
    }
    result = {
        "scope": "One-shot disjoint dual-clock validation; no tracker/P3 outcomes",
        "summary": summary,
        "selector_frozen": True,
        "continue_after_certification": all(gates.values()),
        "alignment_is_natural_p3_ground_truth": False,
        "validation_eligible": False,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
