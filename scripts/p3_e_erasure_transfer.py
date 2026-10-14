#!/usr/bin/env python3
"""Frozen PianoCoRe cross-corpus replication of causal P3-E erasure."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import bayes_from_cost, cosine_cost

ACQUISITION = ROOT / "results/p3_pianocore_p3r_acquisition.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_e_erasure_transfer"
WINDOWS = ((10, 20), (20, 30), (30, 40), (40, 50))
BETA = 8.0
RESTART = 1e-6
BOOTSTRAPS = 20000
SEED = 20260920


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interventions(base, onset, restore):
    erased = base.copy()
    erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
    sham = base.copy()
    sham[onset:restore] += base[onset:restore].mean(axis=1, keepdims=True)
    return erased, sham


def summarize(rows, rng):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["composer"], []).append(row["late_divergence_sec"])
    composer_values = {key: float(np.mean(value)) for key, value in grouped.items()}
    values = np.asarray(list(composer_values.values()))
    boot = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return composer_values, {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "share_gt_0_10": float(np.mean(values > 0.10)),
        "bootstrap_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    transition = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(acquisition["rows"], 1):
        performance_notes = load_notes(ROOT / case["performance_path"])
        score_notes = load_notes(ROOT / case["score_path"])
        performance_duration = max(n[1] for n in performance_notes)
        score_duration = max(n[1] for n in score_notes)
        score_hop = hop * score_duration / performance_duration
        u = notes_to_chroma_stream(performance_notes, 60.0, hop)
        v = notes_to_chroma_stream(score_notes, score_duration, score_hop)
        base = cosine_cost(u, v)
        intact = bayes_from_cost(base, transition, BETA, RESTART)
        for start_sec, restore_sec in WINDOWS:
            onset, restore = int(round(start_sec / hop)), int(round(restore_sec / hop))
            erased_cost, sham_cost = interventions(base, onset, restore)
            erased = bayes_from_cost(erased_cost, transition, BETA, RESTART)
            sham = bayes_from_cost(sham_cost, transition, BETA, RESTART)
            late = slice(restore - int(round(5.0 / hop)), restore)
            pre_equal = bool(np.array_equal(erased[:onset], intact[:onset]) and
                             np.array_equal(sham[:onset], intact[:onset]))
            flat_range = float(np.ptp(erased_cost[onset:restore], axis=1).max())
            sham_equal = bool(np.array_equal(sham, intact))
            distance = np.abs(erased[late] - intact[late]) * score_hop
            rows.append({
                "id": case["id"], "composer": case["composer"],
                "start_sec": start_sec, "restore_sec": restore_sec,
                "score_hop_sec": float(score_hop), "pre_path_equal": pre_equal,
                "erased_max_row_range": flat_range, "sham_path_equal": sham_equal,
                "late_divergence_sec": float(np.mean(distance)),
                "end_divergence_sec": float(abs(erased[restore - 1] - intact[restore - 1]) * score_hop),
            })
        print(f"[{number}/{len(acquisition['rows'])}] {case['id']}")
    composer_values, metrics = summarize(rows, np.random.default_rng(SEED))
    gates = {
        "coverage_30_composers_120_interventions": len(composer_values) == 30 and len(rows) == 120,
        "all_pre_paths_equal": all(r["pre_path_equal"] for r in rows),
        "all_erased_rows_flat": max(r["erased_max_row_range"] for r in rows) <= 1e-12,
        "all_sham_paths_equal": all(r["sham_path_equal"] for r in rows),
        "mean_ge_0_50": metrics["mean"] >= 0.50,
        "median_ge_0_10": metrics["median"] >= 0.10,
        "share_gt_0_10_ge_0_70": metrics["share_gt_0_10"] >= 0.70,
        "bootstrap_lower_gt_0": metrics["bootstrap_95"][0] > 0,
    }
    composer_rows = [{"composer": key, "mean_late_divergence_sec": value}
                     for key, value in sorted(composer_values.items())]
    result = {
        "hypothesis": "P3-E positional-information erasure causes path divergence",
        "scope": "Frozen cross-corpus causal replication on PianoCoRe 30-composer cohort; not natural temporal P3",
        "parameters": {"windows_sec": WINDOWS, "late_outcome_sec": 5.0, "hop_sec": hop,
                       "transition": transition, "emission_beta": BETA,
                       "restart_probability": RESTART, "bootstraps": BOOTSTRAPS, "seed": SEED},
        "counts": {"composers": len(composer_values), "interventions": len(rows)},
        "metrics": metrics, "gates": gates, "transfer_pass": all(gates.values()),
        "p3_e_cross_corpus_supported": all(gates.values()),
        "original_natural_temporal_p3_confirmed": False,
        "provenance": {"acquisition_sha256": digest(ACQUISITION),
                       "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(ROOT / "P3_E_ERASURE_TRANSFER_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "interventions.csv", rows)
    write_csv(OUT / "composer_effects.csv", composer_rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
