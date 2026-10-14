#!/usr/bin/env python3
"""Frozen P3-D partial positional-information dose-response development test."""

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
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import bayes_from_cost, cosine_cost

ACQUISITION = ROOT / "results/p3_pianocore_p3r_acquisition.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_d_dose_response"
WINDOWS = ((10, 20), (20, 30), (30, 40), (40, 50))
LAMBDAS = (1.0, 0.75, 0.5, 0.25, 0.0)
BETA = 8.0
RESTART = 1e-6
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attenuate(base, onset, restore, dose):
    result = base.copy()
    mean = base[onset:restore].mean(axis=1, keepdims=True)
    result[onset:restore] = mean + dose * (base[onset:restore] - mean)
    return result


def transition_matrix(n, weights):
    matrix = np.zeros((n, n))
    for source in range(n):
        for step, weight in enumerate(weights):
            matrix[min(source + step, n - 1), source] += weight
    return matrix


def bayes_trace(cost, transition, beta=BETA, restart=RESTART):
    frames, states = cost.shape
    posterior = np.zeros(states)
    posterior[0] = 1.0
    trace = np.empty((frames, states))
    path = np.empty(frames, dtype=int)
    for frame in range(frames):
        if frame:
            prior = np.zeros(states)
            for step, weight in enumerate(transition):
                if step == 0:
                    prior += weight * posterior
                else:
                    prior[step:] += weight * posterior[:-step]
                    prior[-1] += weight * posterior[-step:].sum()
            posterior = (1 - restart) * prior + restart / states
        likelihood = np.exp(-beta * (cost[frame] - cost[frame].min()))
        posterior *= likelihood
        posterior /= posterior.sum()
        trace[frame] = posterior
        path[frame] = int(np.argmax(posterior))
    return path, trace


def js_rows(left, right):
    middle = 0.5 * (left + right)
    left_term = np.where(left > 0, left * np.log(left / middle), 0.0)
    right_term = np.where(right > 0, right * np.log(right / middle), 0.0)
    return 0.5 * np.sum(left_term, axis=1) + 0.5 * np.sum(right_term, axis=1)


def contrast_residual(base, changed, onset, restore, dose):
    base_range = np.ptp(base[onset:restore], axis=1)
    changed_range = np.ptp(changed[onset:restore], axis=1)
    mask = base_range > 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(changed_range[mask] / base_range[mask] - dose)))


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["composer"], row["lambda"])
        grouped.setdefault(key, []).append(row)
    composer_doses = []
    for (composer, dose), values in sorted(grouped.items()):
        composer_doses.append({
            "composer": composer, "lambda": dose, "attenuation": 1 - dose,
            "map_divergence_sec": float(np.mean([v["map_divergence_sec"] for v in values])),
            "posterior_js": float(np.mean([v["posterior_js"] for v in values])),
        })
    effects = []
    for composer in sorted({r["composer"] for r in composer_doses}):
        values = sorted([r for r in composer_doses if r["composer"] == composer],
                        key=lambda r: r["attenuation"])
        attenuation = [r["attenuation"] for r in values]
        effects.append({
            "composer": composer,
            "map_dose_rho": float(spearmanr(attenuation, [r["map_divergence_sec"] for r in values]).statistic),
            "posterior_dose_rho": float(spearmanr(attenuation, [r["posterior_js"] for r in values]).statistic),
        })
    return composer_doses, effects


def bootstrap(values, rng):
    samples = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


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
        intact_path, intact_trace = bayes_trace(base, transition)
        if not np.array_equal(intact_path, bayes_from_cost(base, transition, BETA, RESTART)):
            raise AssertionError("trace implementation path mismatch")
        for start_sec, restore_sec in WINDOWS:
            onset, restore = int(round(start_sec / hop)), int(round(restore_sec / hop))
            late = slice(restore - int(round(5.0 / hop)), restore)
            for dose in LAMBDAS:
                changed = attenuate(base, onset, restore, dose)
                path, trace = bayes_trace(changed, transition)
                rows.append({
                    "id": case["id"], "composer": case["composer"],
                    "start_sec": start_sec, "restore_sec": restore_sec,
                    "lambda": dose, "score_hop_sec": float(score_hop),
                    "pre_path_equal": bool(np.array_equal(path[:onset], intact_path[:onset])),
                    "lambda1_path_equal": bool(dose != 1.0 or np.array_equal(path, intact_path)),
                    "lambda1_posterior_residual": (0.0 if dose != 1.0 else
                                                    float(np.max(np.abs(trace - intact_trace)))),
                    "contrast_ratio_residual": contrast_residual(base, changed, onset, restore, dose),
                    "map_divergence_sec": float(np.mean(np.abs(path[late] - intact_path[late]) * score_hop)),
                    "posterior_js": float(np.mean(js_rows(trace[late], intact_trace[late]))),
                })
        print(f"[{number}/{len(acquisition['rows'])}] {case['id']}")
    composer_doses, effects = aggregate(rows)
    half = [r for r in composer_doses if r["lambda"] == 0.5]
    half_map = np.asarray([r["map_divergence_sec"] for r in half])
    half_js = np.asarray([r["posterior_js"] for r in half])
    mean_curve = []
    for dose in sorted(LAMBDAS, reverse=True):
        values = [r for r in composer_doses if r["lambda"] == dose]
        mean_curve.append({"lambda": dose, "attenuation": 1 - dose,
                           "map_divergence_sec": float(np.mean([r["map_divergence_sec"] for r in values])),
                           "posterior_js": float(np.mean([r["posterior_js"] for r in values]))})
    map_curve_rho = float(spearmanr([r["attenuation"] for r in mean_curve],
                                    [r["map_divergence_sec"] for r in mean_curve]).statistic)
    js_curve_rho = float(spearmanr([r["attenuation"] for r in mean_curve],
                                   [r["posterior_js"] for r in mean_curve]).statistic)
    map_positive_share = float(np.mean([r["map_dose_rho"] > 0 for r in effects]))
    js_positive_share = float(np.mean([r["posterior_dose_rho"] > 0 for r in effects]))
    rng = np.random.default_rng(SEED)
    map_ci = bootstrap(half_map, rng)
    js_ci = bootstrap(half_js, rng)
    metrics = {
        "mean_curve": mean_curve,
        "half_dose": {"map_mean_sec": float(half_map.mean()), "map_bootstrap_95": map_ci,
                      "map_composer_share_gt_0_02": float(np.mean(half_map > 0.02)),
                      "posterior_js_mean": float(half_js.mean()), "posterior_js_bootstrap_95": js_ci},
        "dose_monotonicity": {"mean_curve_map_rho": map_curve_rho,
                              "mean_curve_posterior_rho": js_curve_rho,
                              "positive_composer_map_rho_share": map_positive_share,
                              "positive_composer_posterior_rho_share": js_positive_share},
    }
    gates = {
        "coverage_30_composers_600_runs": len(effects) == 30 and len(rows) == 600,
        "all_pre_paths_equal": all(r["pre_path_equal"] for r in rows),
        "all_lambda1_paths_equal": all(r["lambda1_path_equal"] for r in rows),
        "all_lambda1_posteriors_equal": max(r["lambda1_posterior_residual"] for r in rows) <= 1e-12,
        "contrast_ratio_residual_le_1e_12": max(r["contrast_ratio_residual"] for r in rows) <= 1e-12,
        "half_map_mean_ge_0_10": half_map.mean() >= 0.10,
        "half_map_bootstrap_lower_gt_0": map_ci[0] > 0,
        "half_map_composer_share_gt_0_02_ge_0_70": np.mean(half_map > 0.02) >= 0.70,
        "half_posterior_js_mean_ge_0_005": half_js.mean() >= 0.005,
        "half_posterior_js_bootstrap_lower_gt_0": js_ci[0] > 0,
        "mean_curve_map_rho_ge_0_90": map_curve_rho >= 0.90,
        "mean_curve_posterior_rho_ge_0_90": js_curve_rho >= 0.90,
        "positive_composer_map_rho_share_ge_0_70": map_positive_share >= 0.70,
        "positive_composer_posterior_rho_share_ge_0_80": js_positive_share >= 0.80,
    }
    result = {
        "hypothesis": "P3-D partial positional-information dose response",
        "scope": "Previously opened PianoCoRe 30-composer development; not independent confirmation",
        "parameters": {"lambdas": LAMBDAS, "windows_sec": WINDOWS, "outcome_final_sec": 5.0,
                       "hop_sec": hop, "transition": transition, "emission_beta": BETA,
                       "restart_probability": RESTART, "bootstraps": BOOTSTRAPS, "seed": SEED},
        "counts": {"composers": len(effects), "runs": len(rows)},
        "metrics": metrics, "gates": {key: bool(value) for key, value in gates.items()},
        "development_pass": bool(all(gates.values())),
        "preserved_validation_authorized": bool(all(gates.values())),
        "natural_temporal_p3_confirmed": False,
        "provenance": {"acquisition_sha256": digest(ACQUISITION), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(ROOT / "P3_D_DOSE_RESPONSE_PROTOCOL.md"),
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
