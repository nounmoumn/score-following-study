#!/usr/bin/env python3
"""One-shot frozen validation of P3-DP on 17 preserved PianoCoRe composers."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_d_dose_response as dose

ACQUISITION = ROOT / "results/p3_dp_pianocore_validation_acquisition.json"
DEVELOPMENT = ROOT / "results/p3_pianocore_p3r_acquisition.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_dp_pianocore_validation"
WINDOWS = ((10, 20), (20, 30), (30, 40), (40, 50))
LAMBDAS = (1.0, 0.75, 0.5, 0.25, 0.0)
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf8"))
    development = json.loads(DEVELOPMENT.read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    transition = tuple(float(x) for x in cfg["bayes"]["transition"])
    validation_composers = {r["composer"] for r in acquisition["rows"]}
    development_composers = {r["composer"] for r in development["rows"]}
    overlap = sorted(validation_composers & development_composers)
    rows = []
    for number, case in enumerate(acquisition["rows"], 1):
        performance_notes = dose.load_notes(ROOT / case["performance_path"])
        score_notes = dose.load_notes(ROOT / case["score_path"])
        performance_duration = max(n[1] for n in performance_notes)
        score_duration = max(n[1] for n in score_notes)
        score_hop = hop * score_duration / performance_duration
        u = dose.notes_to_chroma_stream(performance_notes, 60.0, hop)
        v = dose.notes_to_chroma_stream(score_notes, score_duration, score_hop)
        base = dose.cosine_cost(u, v)
        intact_path, intact_trace = dose.bayes_trace(base, transition)
        for start_sec, restore_sec in WINDOWS:
            onset, restore = int(round(start_sec / hop)), int(round(restore_sec / hop))
            if restore > len(base):
                raise AssertionError("preserved performance lacks a frozen window")
            late = slice(restore - int(round(5.0 / hop)), restore)
            for value in LAMBDAS:
                changed = dose.attenuate(base, onset, restore, value)
                path, trace = dose.bayes_trace(changed, transition)
                rows.append({
                    "id": case["id"], "composer": case["composer"],
                    "start_sec": start_sec, "restore_sec": restore_sec, "lambda": value,
                    "pre_path_equal": bool(np.array_equal(path[:onset], intact_path[:onset])),
                    "lambda1_path_equal": bool(value != 1.0 or np.array_equal(path, intact_path)),
                    "lambda1_posterior_residual": (0.0 if value != 1.0 else
                                                    float(np.max(np.abs(trace - intact_trace)))),
                    "contrast_ratio_residual": dose.contrast_residual(base, changed, onset, restore, value),
                    "posterior_js": float(np.mean(dose.js_rows(trace[late], intact_trace[late]))),
                    "map_divergence_sec_diagnostic": float(np.mean(np.abs(path[late] - intact_path[late]) * score_hop)),
                })
        print(f"[{number}/{len(acquisition['rows'])}] {case['id']}")
    # Reuse the frozen aggregation; rename its MAP field only in output interpretation.
    aggregation_rows = [{**r, "map_divergence_sec": r["map_divergence_sec_diagnostic"]} for r in rows]
    composer_doses, effects = dose.aggregate(aggregation_rows)
    half = [r for r in composer_doses if r["lambda"] == 0.5]
    half_js = np.asarray([r["posterior_js"] for r in half])
    half_map = np.asarray([r["map_divergence_sec"] for r in half])
    curve = []
    for value in sorted(LAMBDAS, reverse=True):
        items = [r for r in composer_doses if r["lambda"] == value]
        curve.append({"lambda": value, "attenuation": 1 - value,
                      "posterior_js": float(np.mean([r["posterior_js"] for r in items])),
                      "map_divergence_sec_diagnostic": float(np.mean([r["map_divergence_sec"] for r in items]))})
    curve_rho = float(spearmanr([r["attenuation"] for r in curve],
                                [r["posterior_js"] for r in curve]).statistic)
    positive_rho_share = float(np.mean([r["posterior_dose_rho"] > 0 for r in effects]))
    rng = np.random.default_rng(SEED)
    boot = rng.choice(half_js, size=(BOOTSTRAPS, len(half_js)), replace=True).mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    metrics = {
        "mean_curve": curve,
        "half_dose": {"posterior_js_mean": float(half_js.mean()),
                      "posterior_js_median": float(np.median(half_js)),
                      "posterior_js_bootstrap_95": ci,
                      "composer_share_gt_0_002": float(np.mean(half_js > 0.002)),
                      "map_mean_sec_diagnostic": float(half_map.mean()),
                      "map_median_sec_diagnostic": float(np.median(half_map))},
        "dose_monotonicity": {"mean_curve_posterior_rho": curve_rho,
                              "positive_composer_posterior_rho_share": positive_rho_share},
    }
    gates = {
        "coverage_17_composers_68_windows_340_runs": len(validation_composers) == 17 and len(rows) == 340,
        "acquisition_complete_and_distinct": acquisition["acquired"] == 17 and acquisition["all_score_performance_distinct"],
        "zero_development_composer_overlap": len(overlap) == 0,
        "all_pre_paths_equal": all(r["pre_path_equal"] for r in rows),
        "all_lambda1_paths_equal": all(r["lambda1_path_equal"] for r in rows),
        "all_lambda1_posteriors_equal": max(r["lambda1_posterior_residual"] for r in rows) <= 1e-12,
        "contrast_ratio_residual_le_1e_12": max(r["contrast_ratio_residual"] for r in rows) <= 1e-12,
        "half_posterior_js_mean_ge_0_010": half_js.mean() >= 0.010,
        "half_posterior_js_median_ge_0_005": np.median(half_js) >= 0.005,
        "half_composer_share_gt_0_002_ge_0_80": np.mean(half_js > 0.002) >= 0.80,
        "half_bootstrap_lower_gt_0_005": ci[0] > 0.005,
        "mean_curve_posterior_rho_ge_0_90": curve_rho >= 0.90,
        "positive_composer_posterior_rho_share_ge_0_80": positive_rho_share >= 0.80,
    }
    result = {
        "hypothesis": "P3-DP partial attenuation causes graded posterior displacement",
        "scope": "One-shot confirmation on preserved PianoCoRe 17-composer split; MAP diagnostic only",
        "parameters": {"lambdas": LAMBDAS, "windows_sec": WINDOWS, "outcome_final_sec": 5.0,
                       "hop_sec": hop, "transition": transition, "emission_beta": dose.BETA,
                       "restart_probability": dose.RESTART, "bootstraps": BOOTSTRAPS, "seed": SEED},
        "counts": {"composers": len(validation_composers), "windows": len(rows) // len(LAMBDAS),
                   "runs": len(rows), "development_composer_overlap": len(overlap)},
        "metrics": metrics, "gates": {k: bool(v) for k, v in gates.items()},
        "validation_pass": bool(all(gates.values())),
        "p3_dp_confirmed": bool(all(gates.values())),
        "p3_d_map_claim_reopened": False,
        "natural_temporal_p3_confirmed": False,
        "provenance": {"acquisition_sha256": digest(ACQUISITION), "development_acquisition_sha256": digest(DEVELOPMENT),
                       "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(ROOT / "P3_DP_PIANOCORE_VALIDATION_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "runs.csv", rows)
    write_csv(OUT / "composer_doses.csv", composer_doses)
    write_csv(OUT / "composer_monotonicity.csv", effects)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

