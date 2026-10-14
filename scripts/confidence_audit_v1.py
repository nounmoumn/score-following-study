"""Frozen, descriptive current-output confidence audit on opened ASAP data."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.confidence_audit import (
    anchor_labels, blocked_maxsim, confidence_trace, summarize, work_weights,
)
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import bayes_from_cost, cosine_cost

CONFIG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/confidence_audit_v1.json")
PROTOCOL = ROOT / "CONFIDENCE_AUDIT_V1_PROTOCOL.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(case, trace, indices, times, truth, ages, cfg, quantiles):
    rows = []
    for k, i in enumerate(indices):
        row = {"case_id": case["case_id"], "work_family": case["work_family"],
               "composer": case["composer"], "performance_time_sec": float(times[k]),
               "available_at_sec": float((i+1)*cfg["performance_hop_sec"]),
               "output_age_sec": float(ages[k]), "truth_score_sec": float(truth[k])}
        row.update({field: float(values[i]) for field, values in trace.items()})
        error = abs(row["position_score_sec"]-row["truth_score_sec"])
        row.update(error_score_sec=error, correct=int(error <= cfg["position_tolerance_score_sec"]+1e-12))
        if row["ambiguity_valid_mass"] < .99 or quantiles is None:
            group = "missing_or_boundary"
        elif quantiles[0] == quantiles[1]:
            group = "degenerate_score_quantiles"
        elif row["ambiguity"] <= quantiles[0]:
            group = "low"
        elif row["ambiguity"] >= quantiles[1]:
            group = "high"
        else:
            group = "middle"
        row["ambiguity_group"] = group
        rows.append(row)
    return rows


def aggregate(rows, cfg):
    w = work_weights(rows)
    common = dict(bins=cfg["reliability_bins"], high=cfg["high_confidence"], clip=cfg["log_clip"])
    result = {"work_equal": summarize(rows, w, **common), "micro": summarize(rows, **common)}
    groups = sorted({r["ambiguity_group"] for r in rows})
    strata = []
    for group in groups:
        mask = np.array([r["ambiguity_group"] == group for r in rows])
        subset = [r for r in rows if r["ambiguity_group"] == group]
        strata.append({"group": group, "total_weight": float(w[mask].sum()),
                       "cases": len({r["case_id"] for r in subset}),
                       "works": len({r["work_family"] for r in subset}),
                       **summarize(subset, w[mask], **common)})
    result["ambiguity_strata"] = strata
    return result


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["out"]
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    source = ROOT / cfg["source_manifest"]
    with source.open(encoding="utf-8-sig", newline="") as f:
        cases = [r for r in csv.DictReader(f) if r["role"] == cfg["role"]]
    if len(cases) != cfg["expected_performances"]:
        raise AssertionError("unexpected cohort count")
    data = ROOT / cfg["asap_root"]
    annotation_path = data / "asap_annotations.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    preflight = []
    for case in cases:
        issues = []
        for field in ("score", "performance"):
            if digest(data / case[field]) != case[field+"_sha256"]:
                issues.append(field+" hash mismatch")
        ann = annotations.get(case["performance"], {})
        if not ann.get("score_and_performance_aligned", False):
            issues.append("annotation not aligned")
        tb = np.asarray(ann.get("performance_beats", []), dtype=float)
        sb = np.asarray(ann.get("midi_score_beats", []), dtype=float)
        if (len(tb) != len(sb) or len(tb) < 2 or not np.isfinite(tb).all()
                or not np.isfinite(sb).all() or np.any(np.diff(tb) <= 0)
                or np.any(np.diff(sb) <= 0)):
            issues.append("invalid/unrepaired annotation clock")
        preflight.append({**case, "annotation_count": len(tb), "issues": issues})
    frozen_paths = [CONFIG, PROTOCOL, Path(__file__), source, annotation_path,
                    ROOT / "src/score_following/confidence_audit.py",
                    ROOT / "src/score_following/features.py",
                    ROOT / "src/score_following/p3_extension.py",
                    ROOT / "src/score_following/observation_evidence.py",
                    ROOT / "tests/test_confidence_audit.py"]
    freeze = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "previously exposed development; no independent validation",
              "python": platform.python_version(), "numpy": np.__version__,
              "config": cfg, "cases": preflight,
              "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in frozen_paths}}
    write_json(out / "freeze.json", freeze)
    write_json(out / "state.json", {"status": "RUNNING", "freeze_sha256": digest(out/"freeze.json")})
    primary, secondary, summaries, failures, score_cache = [], [], [], [], {}
    hp, hs = cfg["performance_hop_sec"], cfg["score_hop_sec"]
    options = dict(score_hop=hs, tolerance=cfg["position_tolerance_score_sec"],
                   transition=cfg["transition"], beta=cfg["beta"], restart=cfg["restart"], clip=cfg["log_clip"])
    with threadpool_limits(limits=cfg["blas_threads"]):
        for number, case in enumerate(preflight, 1):
            case_start = time.perf_counter()
            try:
                if case["issues"]:
                    raise ValueError("; ".join(case["issues"]))
                ann = annotations[case["performance"]]
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    pnotes = load_notes(data / case["performance"])
                    cache_key = case["score_sha256"]
                    if cache_key not in score_cache:
                        snotes = load_notes(data / case["score"])
                        v = notes_to_chroma_stream(snotes, max(n[1] for n in snotes), hs)
                        amb, silent = blocked_maxsim(v, int(round(cfg["ambiguity_window_score_sec"]/hs)),
                                                    int(round(cfg["ambiguity_exclude_score_sec"]/hs)),
                                                    cfg["matrix_block_rows"])
                        finite = amb[np.isfinite(amb)]
                        quantiles = np.quantile(finite, cfg["ambiguity_quantiles"]).tolist() if len(finite) else None
                        score_cache[cache_key] = v, amb, quantiles, silent
                v, amb, quantiles, silent = score_cache[cache_key]
                duration = max(n[1] for n in pnotes)
                frames = int(np.floor(min(cfg["replay_prefix_sec"], duration)/hp))
                if frames < 2:
                    raise ValueError("not enough complete performance frames")
                u = notes_to_chroma_stream(pnotes, frames*hp, hp)[:frames]
                # Reading a future note-off from a file must not change past completed audio activity.
                cut = min(frames, 73)
                cutoff = cut*hp
                truncated = [(s, min(e, cutoff), p) for s,e,p in pnotes if s < cutoff]
                prefix = notes_to_chroma_stream(truncated, cutoff, hp)[:cut]
                if not np.allclose(u[:cut], prefix, rtol=0, atol=1e-12):
                    raise AssertionError("future note-off altered past features")
                cost = cosine_cost(u, v)
                trace = confidence_trace(cost, amb, **options)
                reference = bayes_from_cost(cost, cfg["transition"], cfg["beta"], cfg["restart"])
                if not np.array_equal(reference, trace["map_index"]):
                    raise AssertionError("original Bayes path mismatch")
                trace_prefix = confidence_trace(cost[:cut], amb, **options)
                if any(not np.array_equal(trace[k][:cut], trace_prefix[k]) for k in trace):
                    raise AssertionError("future frames changed earlier inference")
                labels = anchor_labels(trace, ann["performance_beats"], ann["midi_score_beats"],
                                       hop=hp, tolerance=cfg["position_tolerance_score_sec"])
                rows = make_rows(case, trace, labels["indices"], labels["times"], labels["truth"],
                                 labels["ages"], cfg, quantiles)
                if not rows:
                    raise ValueError("no evaluable original beat anchors")
                times = (np.arange(frames)+1)*hp
                inside = (times >= ann["performance_beats"][0]) & (times <= ann["performance_beats"][-1])
                idx = np.flatnonzero(inside)
                truth = np.interp(times[idx], ann["performance_beats"], ann["midi_score_beats"])
                extra = make_rows(case, trace, idx, times[idx], truth, np.zeros(len(idx)), cfg, quantiles)
                primary.extend(rows)
                secondary.extend(extra)
                stats = summarize(rows)
                summary = {"case_id": case["case_id"], "work_family": case["work_family"],
                           "frames": frames, "score_states": len(v), "primary_beats": len(rows),
                           "secondary_frames": len(extra), "replay_end_sec": frames*hp,
                           "output_age_max_sec": float(max(r["output_age_sec"] for r in rows)),
                           "undefined_score_states": int(np.sum(~np.isfinite(amb))),
                           "silent_contexts": silent, "score_ambiguity_quantiles": quantiles,
                           "prefix_feature_equal": True, "prefix_filter_equal": True,
                           "original_bayes_path_equal": True,
                           "warnings": sorted({str(w.message) for w in recorded}),
                           "runtime_sec": time.perf_counter()-case_start,
                           **{k:v for k,v in stats.items() if k != "reliability"}}
                summaries.append(summary)
                write_csv(out / (case["case_id"]+"_beats.csv"), rows)
                write_csv(out / (case["case_id"]+"_frames.csv"), extra)
                print(f"[{number}/{len(cases)}] {case['case_id']} beats={len(rows)} accuracy={stats['accuracy']:.3f} confidence={stats['mean_confidence']:.3f}", flush=True)
            except Exception as error:
                failure = {"case_id": case["case_id"], "error": str(error), "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(out / (case["case_id"]+"_failure.json"), failure)
                print(f"[{number}/{len(cases)}] {case['case_id']} FAILED: {error}", flush=True)
    if any(digest(ROOT/path) != value for path,value in freeze["files"].items()):
        raise AssertionError("source changed during execution")
    result = {"scope": freeze["scope"], "fresh_holdout": False, "calibrator_fitted": False,
              "expected_cases": len(cases), "completed_cases": len(summaries), "failures": failures,
              "completed_work_families": len({r["work_family"] for r in summaries}),
              "primary": aggregate(primary, cfg) if primary else None,
              "secondary_interpolated": aggregate(secondary, cfg) if secondary else None,
              "case_summaries": summaries, "wall_sec": time.perf_counter()-started,
              "freeze_sha256": digest(out/"freeze.json"),
              "source_hashes_unchanged": True,
              "decision": "descriptive_audit_complete_no_calibration_efficacy_claim" if not failures else "incomplete_coverage_inspect_failures"}
    write_json(out/"result.json", result)
    write_csv(out/"primary_beats.csv", primary)
    write_csv(out/"secondary_frames.csv", secondary)
    write_json(out/"state.json", {"status": "COMPLETE" if not failures else "INCOMPLETE",
                                  "result_sha256": digest(out/"result.json"), "decision": result["decision"]})
    print(json.dumps({"completed":len(summaries),"failures":len(failures),
                      "primary":result["primary"]["work_equal"] if primary else None}, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
