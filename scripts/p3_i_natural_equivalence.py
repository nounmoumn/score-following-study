#!/usr/bin/env python3
"""Frozen KRAISLER development test of natural information equivalence."""

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
OUT = ROOT / "results/p3_i_natural_equivalence"
LANDMARKS_SEC = tuple(range(10, 46, 5))
WINDOW_FRAMES = 20
FINAL_FRAMES = 5
BOOTSTRAPS = 20000
PERMUTATIONS = 2000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def likelihood_information(cost, beta=dose.BETA):
    likelihood = np.exp(-beta * (cost - cost.min(axis=1, keepdims=True)))
    likelihood /= likelihood.sum(axis=1, keepdims=True)
    log_likelihood = np.zeros_like(likelihood)
    np.log(likelihood, out=log_likelihood, where=likelihood > 0)
    entropy = -np.sum(likelihood * log_likelihood, axis=1)
    return 1.0 - entropy / np.log(cost.shape[1])


def replay(initial, cost, transition, beta=dose.BETA, restart=dose.RESTART):
    posterior = initial.copy()
    trace = []
    states = len(initial)
    for row in cost:
        prior = np.zeros(states)
        for step, weight in enumerate(transition):
            if step == 0:
                prior += weight * posterior
            else:
                prior[step:] += weight * posterior[:-step]
                prior[-1] += weight * posterior[-step:].sum()
        posterior = (1 - restart) * prior + restart / states
        likelihood = np.exp(-beta * (row - row.min()))
        posterior *= likelihood
        posterior /= posterior.sum()
        trace.append(posterior.copy())
    return np.asarray(trace)


def summarize(rows, information_override=None):
    override = None if information_override is None else np.asarray(information_override)
    effects = []
    for track in sorted({r["track"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["track"] == track]
        information = (np.asarray([rows[i]["information"] for i in indices])
                       if override is None else override[indices])
        response = np.asarray([rows[i]["response_js"] for i in indices])
        rho = float(spearmanr(information, response).statistic)
        order = np.argsort(information)
        quartile = max(1, len(indices) // 4)
        contrast = float(np.mean(response[order[-quartile:]]) - np.mean(response[order[:quartile]]))
        effects.append({"track": track, "rho": rho, "quartile_contrast": contrast,
                        "landmarks": len(indices)})
    return effects


def shifted_information(rows, rng):
    values = np.asarray([r["information"] for r in rows])
    shifted = values.copy()
    for track in sorted({r["track"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["track"] == track]
        n = len(indices)
        allowed = [s for s in range(1, n) if min(s, n-s) >= 2]
        shifted[indices] = np.roll(values[indices], int(rng.choice(allowed)))
    return shifted


def bootstrap(values, rng):
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    from p3_kraisler_signal_run import load_case

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [r for r in manifest["rows"] if r["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    transition = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, duration = load_case(case, hop)
        base = dose.cosine_cost(u, v)
        _, actual_trace = dose.bayes_trace(base, transition)
        for time_sec in LANDMARKS_SEC:
            end = int(round(time_sec / hop))
            start = end - WINDOW_FRAMES
            initial = actual_trace[start - 1]
            natural = replay(initial, base[start:end], transition)
            erased_cost = base[start:end].mean(axis=1, keepdims=True)
            erased_cost = np.broadcast_to(erased_cost, base[start:end].shape)
            erased = replay(initial, erased_cost, transition)
            residual = float(np.max(np.abs(natural - actual_trace[start:end])))
            rows.append({"track": case["track"], "composer": case["composer"],
                         "time_sec": time_sec,
                         "information": float(np.mean(likelihood_information(base[start:end]))),
                         "response_js": float(np.mean(dose.js_rows(natural[-FINAL_FRAMES:], erased[-FINAL_FRAMES:]))),
                         "natural_replay_residual": residual})
        print(f"[{number}/{len(cases)}] track={case['track']}")
    effects = summarize(rows)
    rho = np.asarray([r["rho"] for r in effects])
    contrast = np.asarray([r["quartile_contrast"] for r in effects])
    rng = np.random.default_rng(SEED)
    rho_ci = bootstrap(rho, rng)
    contrast_ci = bootstrap(contrast, rng)
    null = []
    for _ in range(PERMUTATIONS):
        null.append(float(np.mean([r["rho"] for r in summarize(rows, shifted_information(rows, rng))])))
    observed = float(rho.mean())
    permutation_p = float((1 + np.sum(np.asarray(null) >= observed)) / (PERMUTATIONS + 1))
    metrics = {"mean_track_rho": observed, "median_track_rho": float(np.median(rho)),
               "positive_track_rho_share": float(np.mean(rho > 0)), "rho_bootstrap_95": rho_ci,
               "mean_quartile_contrast": float(contrast.mean()),
               "quartile_contrast_bootstrap_95": contrast_ci,
               "circular_shift_p": permutation_p, "null_mean_rho_p95": float(np.percentile(null, 95))}
    gates = {
        "coverage_13_tracks_104_landmarks": len(effects) == 13 and len(rows) == 104,
        "natural_replay_residual_le_1e_12": max(r["natural_replay_residual"] for r in rows) <= 1e-12,
        "mean_track_rho_ge_0_50": rho.mean() >= 0.50,
        "median_track_rho_ge_0_40": np.median(rho) >= 0.40,
        "positive_track_rho_share_ge_0_80": np.mean(rho > 0) >= 0.80,
        "rho_bootstrap_lower_gt_0": rho_ci[0] > 0,
        "mean_quartile_contrast_ge_0_020": contrast.mean() >= 0.020,
        "quartile_contrast_bootstrap_lower_gt_0": contrast_ci[0] > 0,
        "circular_shift_p_le_0_05": permutation_p <= 0.05,
    }
    result = {
        "hypothesis": "P3-I natural positional-information equivalence",
        "scope": "KRAISLER endpoint development; unmodified natural inputs, no GT or future outcome",
        "parameters": {"landmarks_sec": LANDMARKS_SEC, "history_sec": 2.0, "response_final_sec": 0.5,
                       "transition": transition, "emission_beta": dose.BETA,
                       "restart_probability": dose.RESTART, "bootstraps": BOOTSTRAPS,
                       "permutations": PERMUTATIONS, "seed": SEED},
        "counts": {"tracks": len(effects), "landmarks": len(rows)},
        "metrics": metrics, "gates": {k: bool(v) for k, v in gates.items()},
        "development_pass": bool(all(gates.values())),
        "pianocore_transfer_authorized": bool(all(gates.values())),
        "future_drift_claim": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(ROOT / "P3_I_NATURAL_EQUIVALENCE_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "landmarks.csv", rows)
    write_csv(OUT / "track_effects.csv", effects)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

