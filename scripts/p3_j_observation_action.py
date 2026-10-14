#!/usr/bin/env python3
"""Numerically audit the exact P3-J natural observation-action identity."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_d_dose_response as dose

MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_J_OBSERVATION_ACTION_PROTOCOL.md"
OUT = ROOT / "results/p3_j_observation_action"
LANDMARKS_SEC = tuple(range(10, 46, 5))
WINDOW_FRAMES = 20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition(values, weights):
    """Column-stochastic tracker transition; accepts probability or signed mass."""
    output = np.zeros_like(values)
    for step, weight in enumerate(weights):
        if step == 0:
            output += weight * values
        else:
            output[step:] += weight * values[:-step]
            output[-1] += weight * values[-step:].sum()
    return output


def update(prior, cost_row, beta=dose.BETA):
    likelihood = np.exp(-beta * (cost_row - cost_row.min()))
    posterior = prior * likelihood
    return posterior / posterior.sum()


def decompose(initial, cost, weights, restart=dose.RESTART):
    intact = initial.copy()
    erased = initial.copy()
    reconstructed = np.zeros_like(initial)
    rows = []
    for cost_row in cost:
        intact_prior = (1.0 - restart) * transition(intact, weights) + restart / len(initial)
        intact = update(intact_prior, cost_row)
        action = intact - intact_prior
        erased = (1.0 - restart) * transition(erased, weights) + restart / len(initial)
        reconstructed = (1.0 - restart) * transition(reconstructed, weights) + action
        rows.append({
            "intact": intact.copy(),
            "erased": erased.copy(),
            "action": action.copy(),
            "reconstructed": reconstructed.copy(),
        })
    return rows


def main():
    from p3_kraisler_signal_run import load_case

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    audit = []
    for number, case in enumerate(cases, 1):
        u, v, _, _ = load_case(case, hop)
        cost = dose.cosine_cost(u, v)
        _, actual = dose.bayes_trace(cost, weights)
        for time_sec in LANDMARKS_SEC:
            end = int(round(time_sec / hop))
            start = end - WINDOW_FRAMES
            rows = decompose(actual[start - 1], cost[start:end], weights)
            intact = np.asarray([row["intact"] for row in rows])
            direct = np.asarray([row["intact"] - row["erased"] for row in rows])
            reconstructed = np.asarray([row["reconstructed"] for row in rows])
            actions = np.asarray([row["action"] for row in rows])
            audit.append({
                "track": case["track"],
                "time_sec": time_sec,
                "intact_replay_residual": float(np.max(np.abs(intact - actual[start:end]))),
                "decomposition_residual": float(np.max(np.abs(direct - reconstructed))),
                "action_mass_residual": float(np.max(np.abs(actions.sum(axis=1)))),
                "final_total_variation": float(0.5 * np.abs(direct[-1]).sum()),
                "mean_action_total_variation": float(np.mean(0.5 * np.abs(actions).sum(axis=1))),
            })
        print(f"[{number}/{len(cases)}] track={case['track']}")

    maxima = {
        "intact_replay_residual": max(row["intact_replay_residual"] for row in audit),
        "decomposition_residual": max(row["decomposition_residual"] for row in audit),
        "action_mass_residual": max(row["action_mass_residual"] for row in audit),
    }
    gates = {
        "coverage_13_tracks_104_windows": len(cases) == 13 and len(audit) == 104,
        "intact_replay_residual_le_1e_12": maxima["intact_replay_residual"] <= 1e-12,
        "decomposition_residual_le_1e_12": maxima["decomposition_residual"] <= 1e-12,
        "action_mass_residual_le_1e_12": maxima["action_mass_residual"] <= 1e-12,
    }
    result = {
        "hypothesis": "P3-J prior-conditioned observation-action decomposition",
        "scope": "exact identity plus natural KRAISLER implementation audit; no future outcome",
        "counts": {"tracks": len(cases), "windows": len(audit),
                   "frames": len(audit) * WINDOW_FRAMES},
        "max_residuals": maxima,
        "descriptive": {
            "mean_final_total_variation": float(np.mean([r["final_total_variation"] for r in audit])),
            "mean_frame_action_total_variation": float(np.mean([r["mean_action_total_variation"] for r in audit])),
        },
        "gates": {key: bool(value) for key, value in gates.items()},
        "identity_confirmed": bool(all(gates.values())),
        "future_drift_claim": False,
        "provenance": {
            "manifest_sha256": digest(MANIFEST),
            "config_sha256": digest(CONFIG),
            "protocol_sha256": digest(PROTOCOL),
            "script_sha256": digest(Path(__file__)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "windows.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

