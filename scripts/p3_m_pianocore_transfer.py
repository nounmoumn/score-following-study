#!/usr/bin/env python3
"""Frozen cross-corpus PianoCoRe transfer of P3-M ambiguity -> weakness."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import evidence_at, score_ambiguity_at, score_window_features

ACQUISITION = ROOT / "results/p3_pianocore_p3r_acquisition.json"
OUT = ROOT / "results/p3_m_pianocore_transfer"
HOP = 0.1
WINDOW = 20
EXCLUDE = 20
RADIUS = 200
BOOTSTRAPS = 10000
SIGN_FLIPS = 19999
PERMUTATIONS = 2000
SEED = 20260920


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alignment_clock(path):
    alignment = np.load(path)
    performance = np.asarray(alignment["perf_times"][:, 0], dtype=float)
    score = np.asarray(alignment["score_times"][:, 0], dtype=float)
    order = np.argsort(performance, kind="stable")
    performance, score = performance[order], score[order]
    unique = np.unique(performance)
    score_median = np.asarray([np.median(score[performance == t]) for t in unique])
    finite = np.isfinite(unique) & np.isfinite(score_median)
    return unique[finite], score_median[finite]


def measure_case(case):
    performance_notes = load_notes(ROOT / case["performance_path"])
    score_notes = load_notes(ROOT / case["score_path"])
    performance_end = max(n[1] for n in performance_notes)
    score_end = max(n[1] for n in score_notes)
    u = notes_to_chroma_stream(performance_notes, performance_end, HOP)
    v = notes_to_chroma_stream(score_notes, score_end, HOP)
    features = score_window_features(v, WINDOW)
    anchor_p, anchor_s = alignment_clock(ROOT / case["alignment_path"])
    rows = []
    exclusions = []
    for time_sec in np.arange(20.0, 75.001, 5.0):
        frame = int(round(time_sec / HOP)) - 1
        if frame >= len(u) or time_sec - 2.0 < anchor_p[0] or time_sec > anchor_p[-1]:
            exclusions.append({"id": case["id"], "time_sec": time_sec, "reason": "coverage"})
            continue
        live_times = np.arange(frame - WINDOW + 1, frame + 1) * HOP
        truth = np.interp(live_times, anchor_p, anchor_s) / HOP
        completed_end = truth[-1]
        try:
            evidence = evidence_at(u, v, truth, WINDOW - 1, window=WINDOW,
                                   exclude=EXCLUDE, radius=RADIUS,
                                   frontier=int(round(completed_end)), band=RADIUS + 1)
            ambiguity = score_ambiguity_at(features, completed_end, WINDOW, EXCLUDE)
            weakness = evidence["weakness_local"]
        except (ValueError, IndexError) as error:
            exclusions.append({"id": case["id"], "time_sec": time_sec,
                               "reason": type(error).__name__})
            continue
        if not np.isfinite(ambiguity) or not np.isfinite(weakness):
            exclusions.append({"id": case["id"], "time_sec": time_sec, "reason": "nonfinite"})
            continue
        rows.append({"id": case["id"], "composer": case["composer"],
                     "time_sec": float(time_sec), "score_time_sec": float(completed_end * HOP),
                     "ambiguity": float(ambiguity), "weakness_local": float(weakness)})
    return rows, exclusions


def composer_effects(rows, ambiguity_override=None):
    effects = []
    composers = sorted({r["composer"] for r in rows})
    override = None if ambiguity_override is None else np.asarray(ambiguity_override)
    for composer in composers:
        indices = [i for i, r in enumerate(rows) if r["composer"] == composer]
        ambiguity = (np.asarray([rows[i]["ambiguity"] for i in indices])
                     if override is None else override[indices])
        weakness = np.asarray([rows[i]["weakness_local"] for i in indices])
        if len(indices) < 8 or np.std(ambiguity) == 0 or np.std(weakness) == 0:
            continue
        effects.append({"composer": composer, "landmarks": len(indices),
                        "rho": float(spearmanr(ambiguity, weakness).statistic)})
    return effects


def shifted_ambiguity(rows, rng):
    values = np.asarray([r["ambiguity"] for r in rows])
    shifted = values.copy()
    for composer in sorted({r["composer"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["composer"] == composer]
        n = len(indices)
        if n < 2:
            continue
        allowed = [s for s in range(1, n) if min(s, n - s) >= min(3, n // 2)] or list(range(1, n))
        shifted[indices] = np.roll(values[indices], int(rng.choice(allowed)))
    return shifted


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf8"))
    rows, exclusions = [], []
    for number, case in enumerate(acquisition["rows"], 1):
        measured, rejected = measure_case(case)
        rows.extend(measured)
        exclusions.extend(rejected)
        print(f"[{number}/{len(acquisition['rows'])}] {case['id']} landmarks={len(measured)}")
    effects = composer_effects(rows)
    values = np.asarray([r["rho"] for r in effects])
    observed = float(values.mean())
    rng = np.random.default_rng(SEED)
    boot = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIPS, len(values)))
    sign_p = float((1 + np.sum((signs * values).mean(axis=1) >= observed - 1e-14)) / (SIGN_FLIPS + 1))
    null = []
    for _ in range(PERMUTATIONS):
        shifted = composer_effects(rows, shifted_ambiguity(rows, rng))
        null.append(float(np.mean([r["rho"] for r in shifted])))
    temporal_p = float((1 + np.sum(np.asarray(null) >= observed)) / (PERMUTATIONS + 1))
    gates = {
        "eligible_composers_ge_25": len(effects) >= 25,
        "landmarks_ge_250": len(rows) >= 250,
        "mean_rho_ge_0_15": observed >= 0.15,
        "positive_composer_share_ge_0_60": float(np.mean(values > 0)) >= 0.60,
        "bootstrap_lower_gt_0": float(np.percentile(boot, 2.5)) > 0,
        "sign_flip_p_le_0_05": sign_p <= 0.05,
        "temporal_shift_p_le_0_05": temporal_p <= 0.05,
    }
    result = {
        "hypothesis": "P3-M structural ambiguity weakens positional evidence",
        "scope": "Previously opened PianoCoRe development; new endpoint, cross-corpus corroboration, not pristine confirmation",
        "counts": {"requested_composers": len(acquisition["rows"]), "eligible_composers": len(effects),
                   "landmarks": len(rows), "excluded_landmarks": len(exclusions)},
        "metrics": {"mean_rho": observed, "positive_composer_share": float(np.mean(values > 0)),
                    "composer_bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "sign_flip_p": sign_p, "temporal_shift_p": temporal_p,
                    "temporal_null_p95": float(np.percentile(null, 95))},
        "gates": gates, "transfer_pass": all(gates.values()),
        "preserved_validation_authorized": all(gates.values()),
        "original_temporal_p3_confirmed": False,
        "provenance": {"acquisition_sha256": digest(ACQUISITION),
                       "protocol_sha256": digest(ROOT / "P3_M_PIANOCORE_TRANSFER_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "landmarks.csv", rows)
    write_csv(OUT / "composer_effects.csv", effects)
    write_csv(OUT / "exclusions.csv", exclusions, ["id", "time_sec", "reason"])
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

