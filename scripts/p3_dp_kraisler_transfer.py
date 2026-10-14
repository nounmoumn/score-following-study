#!/usr/bin/env python3
"""Frozen KRAISLER transfer of P3-DP posterior dose response."""

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

MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_dp_kraisler_transfer"
WINDOWS = ((10, 20), (20, 30), (30, 40), (40, 50))
LAMBDAS = (1.0, 0.75, 0.5, 0.25, 0.0)
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["track"], row["lambda"]), []).append(row)
    track_doses = []
    for (track, value), items in sorted(grouped.items()):
        track_doses.append({
            "track": track, "lambda": value, "attenuation": 1 - value,
            "posterior_js": float(np.mean([x["posterior_js"] for x in items])),
            "map_divergence_sec": float(np.mean([x["map_divergence_sec"] for x in items])),
            "windows": len(items),
        })
    monotonicity = []
    for track in sorted({r["track"] for r in track_doses}):
        items = sorted([r for r in track_doses if r["track"] == track], key=lambda r: r["attenuation"])
        monotonicity.append({
            "track": track,
            "posterior_dose_rho": float(spearmanr([r["attenuation"] for r in items],
                                                  [r["posterior_js"] for r in items]).statistic),
            "map_dose_rho_diagnostic": float(spearmanr([r["attenuation"] for r in items],
                                                        [r["map_divergence_sec"] for r in items]).statistic),
        })
    return track_doses, monotonicity


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    from p3_kraisler_signal_run import load_case
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    transition = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows = []
    complete_windows = 0
    for number, case in enumerate(cases, 1):
        u, v, score_hop, duration = load_case(case, hop)
        base = dose.cosine_cost(u, v)
        intact_path, intact_trace = dose.bayes_trace(base, transition)
        used = 0
        for start_sec, restore_sec in WINDOWS:
            onset, restore = int(round(start_sec / hop)), int(round(restore_sec / hop))
            if restore > len(base):
                continue
            used += 1
            late = slice(restore - int(round(5.0 / hop)), restore)
            for value in LAMBDAS:
                changed = dose.attenuate(base, onset, restore, value)
                path, trace = dose.bayes_trace(changed, transition)
                rows.append({
                    "track": case["track"], "composer": case["composer"],
                    "start_sec": start_sec, "restore_sec": restore_sec, "lambda": value,
                    "pre_path_equal": bool(np.array_equal(path[:onset], intact_path[:onset])),
                    "lambda1_path_equal": bool(value != 1.0 or np.array_equal(path, intact_path)),
                    "lambda1_posterior_residual": (0.0 if value != 1.0 else
                                                    float(np.max(np.abs(trace - intact_trace)))),
                    "contrast_ratio_residual": dose.contrast_residual(base, changed, onset, restore, value),
                    "posterior_js": float(np.mean(dose.js_rows(trace[late], intact_trace[late]))),
                    "map_divergence_sec": float(np.mean(np.abs(path[late] - intact_path[late]) * score_hop)),
                })
        complete_windows += used
        print(f"[{number}/{len(cases)}] track={case['track']} windows={used}")
    track_doses, monotonicity = aggregate(rows)
    half = [r for r in track_doses if r["lambda"] == 0.5]
    half_js = np.asarray([r["posterior_js"] for r in half])
    half_map = np.asarray([r["map_divergence_sec"] for r in half])
    curve = []
    for value in sorted(LAMBDAS, reverse=True):
        items = [r for r in track_doses if r["lambda"] == value]
        curve.append({"lambda": value, "attenuation": 1 - value,
                      "posterior_js": float(np.mean([r["posterior_js"] for r in items])),
                      "map_divergence_sec_diagnostic": float(np.mean([r["map_divergence_sec"] for r in items]))})
    curve_rho = float(spearmanr([r["attenuation"] for r in curve],
                                [r["posterior_js"] for r in curve]).statistic)
    positive_rho_share = float(np.mean([r["posterior_dose_rho"] > 0 for r in monotonicity]))
    rng = np.random.default_rng(SEED)
    boot = rng.choice(half_js, size=(BOOTSTRAPS, len(half_js)), replace=True).mean(axis=1)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    metrics = {
        "mean_curve": curve,
        "half_dose": {"posterior_js_mean": float(half_js.mean()),
                      "posterior_js_median": float(np.median(half_js)),
                      "posterior_js_bootstrap_95": ci,
                      "track_share_gt_0_002": float(np.mean(half_js > 0.002)),
                      "map_mean_sec_diagnostic": float(half_map.mean()),
                      "map_median_sec_diagnostic": float(np.median(half_map))},
        "dose_monotonicity": {"mean_curve_posterior_rho": curve_rho,
                              "positive_track_posterior_rho_share": positive_rho_share},
    }
    gates = {
        "coverage_tracks_ge_12_windows_ge_45_runs_ge_225": len(cases) >= 12 and complete_windows >= 45 and len(rows) >= 225,
        "all_pre_paths_equal": all(r["pre_path_equal"] for r in rows),
        "all_lambda1_paths_equal": all(r["lambda1_path_equal"] for r in rows),
        "all_lambda1_posteriors_equal": max(r["lambda1_posterior_residual"] for r in rows) <= 1e-12,
        "contrast_ratio_residual_le_1e_12": max(r["contrast_ratio_residual"] for r in rows) <= 1e-12,
        "half_posterior_js_mean_ge_0_010": half_js.mean() >= 0.010,
        "half_posterior_js_median_ge_0_005": np.median(half_js) >= 0.005,
        "half_track_share_gt_0_002_ge_0_80": np.mean(half_js > 0.002) >= 0.80,
        "half_bootstrap_lower_gt_0_005": ci[0] > 0.005,
        "mean_curve_posterior_rho_ge_0_90": curve_rho >= 0.90,
        "positive_track_posterior_rho_share_ge_0_80": positive_rho_share >= 0.80,
    }
    result = {
        "hypothesis": "P3-DP partial attenuation causes graded posterior displacement",
        "scope": "Frozen endpoint-specific KRAISLER cross-corpus transfer; MAP diagnostic only",
        "parameters": {"lambdas": LAMBDAS, "windows_sec": WINDOWS, "outcome_final_sec": 5.0,
                       "hop_sec": hop, "transition": transition, "emission_beta": dose.BETA,
                       "restart_probability": dose.RESTART, "bootstraps": BOOTSTRAPS, "seed": SEED},
        "counts": {"eligible_tracks": len(cases), "complete_windows": complete_windows, "runs": len(rows)},
        "metrics": metrics, "gates": {k: bool(v) for k, v in gates.items()},
        "transfer_pass": bool(all(gates.values())),
        "preserved_validation_authorized": bool(all(gates.values())),
        "p3_d_map_claim_reopened": False,
        "natural_temporal_p3_confirmed": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(ROOT / "P3_DP_KRAISLER_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "runs.csv", rows)
    write_csv(OUT / "track_doses.csv", track_doses)
    write_csv(OUT / "track_monotonicity.csv", monotonicity)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
