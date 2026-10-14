#!/usr/bin/env python3
"""Frozen P3-P healthy-state temporal erasure experiment on ASAP."""

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
import p3_i_natural_equivalence as natural
import p3_o_balance_transfer as source
from p3_short_signal import sustained_start

PROTOCOL = ROOT / "P3_P_HEALTHY_ERASURE_PROTOCOL.md"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_p_healthy_erasure"
EXPOSURE = 20
CONFIRM = 4
FINAL = 5
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_risk(trace, truth, score_hop, cap=5.0):
    states = np.arange(trace.shape[1])
    return np.asarray([posterior @ np.minimum(np.abs(states - x) * score_hop, cap)
                       for posterior, x in zip(trace, truth)])


def aggregate(rows):
    effects = []
    for family in sorted({row["family"] for row in rows}):
        group = [row for row in rows if row["family"] == family]
        effects.append({
            "family": family,
            "risk_effect": float(np.mean([row["risk_effect"] for row in group])),
            "map_effect": float(np.mean([row["map_effect"] for row in group])),
            "incidence_effect": float(np.mean([row["erased_event"] - row["intact_event"] for row in group])),
        })
    return effects


def ci(values, rng):
    values = np.asarray(values)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    cases = source.read_csv(source.MANIFEST)
    annotations = json.loads(source.ANNOTATIONS.read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth = source.load_case(case, annotations, hop)
        cost = dose.cosine_cost(u, v)
        _, trace = dose.bayes_trace(cost, weights)
        intact_path = trace.argmax(axis=1)
        intact_error = np.abs(intact_path - truth) * score_hop
        for time_sec in source.LANDMARKS_SEC:
            end = int(round(time_sec / hop))
            if (end + EXPOSURE + CONFIRM >= len(cost)
                    or np.max(intact_error[end - 20:end]) > .5):
                continue
            stop = end + EXPOSURE + CONFIRM
            initial = trace[end - 1]
            intact = natural.replay(initial, cost[end:stop], weights)
            erased_cost = np.broadcast_to(cost[end:stop].mean(axis=1, keepdims=True), cost[end:stop].shape)
            erased = natural.replay(initial, erased_cost, weights)
            local_truth = truth[end:stop]
            intact_risk = expected_risk(intact, local_truth, score_hop)
            erased_risk = expected_risk(erased, local_truth, score_hop)
            intact_local_error = np.abs(intact.argmax(axis=1) - local_truth) * score_hop
            erased_local_error = np.abs(erased.argmax(axis=1) - local_truth) * score_hop
            intact_onset = sustained_start(intact_local_error, 0, EXPOSURE)
            erased_onset = sustained_start(erased_local_error, 0, EXPOSURE)
            rows.append({
                "family": case["work_family"], "composer": case["composer"], "time_sec": time_sec,
                "risk_effect": float(np.mean(erased_risk[EXPOSURE-FINAL:EXPOSURE]
                                             - intact_risk[EXPOSURE-FINAL:EXPOSURE])),
                "map_effect": float(np.mean(erased_local_error[EXPOSURE-FINAL:EXPOSURE]
                                            - intact_local_error[EXPOSURE-FINAL:EXPOSURE])),
                "intact_event": int(intact_onset is not None),
                "erased_event": int(erased_onset is not None),
                "intact_replay_residual": float(np.max(np.abs(intact - trace[end:stop]))),
                "erased_row_range": float(np.max(np.ptp(erased_cost, axis=1))),
            })
        print(f"[{number}/{len(cases)}] {case['work_family']}")

    effects = aggregate(rows)
    risk = np.asarray([row["risk_effect"] for row in effects])
    map_effect = np.asarray([row["map_effect"] for row in effects])
    incidence = np.asarray([row["incidence_effect"] for row in effects])
    rng = np.random.default_rng(SEED)
    metrics = {
        "mean_family_risk_effect": float(risk.mean()), "risk_effect_bootstrap_95": ci(risk, rng),
        "positive_family_risk_share": float(np.mean(risk > 0)),
        "mean_family_map_effect": float(map_effect.mean()), "map_effect_bootstrap_95": ci(map_effect, rng),
        "mean_family_incidence_effect": float(incidence.mean()),
        "incidence_effect_bootstrap_95": ci(incidence, rng),
        "intact_event_rate": float(np.mean([row["intact_event"] for row in rows])),
        "erased_event_rate": float(np.mean([row["erased_event"] for row in rows])),
    }
    gates = {
        "coverage": len(effects) == 25 and len(rows) >= 250,
        "intact_replay_residual_le_1e_12": max(row["intact_replay_residual"] for row in rows) <= 1e-12,
        "erased_rows_flat": max(row["erased_row_range"] for row in rows) <= 1e-12,
        "risk_effect_ge_0_05": risk.mean() >= .05,
        "risk_ci_lower_gt_0": metrics["risk_effect_bootstrap_95"][0] > 0,
        "risk_positive_share_ge_0_70": np.mean(risk > 0) >= .70,
        "map_effect_ge_0_05": map_effect.mean() >= .05,
        "map_ci_lower_gt_0": metrics["map_effect_bootstrap_95"][0] > 0,
        "incidence_effect_ge_0_03": incidence.mean() >= .03,
        "incidence_ci_lower_gt_0": metrics["incidence_effect_bootstrap_95"][0] > 0,
    }
    result = {
        "hypothesis": "P3-P healthy-state temporal erasure",
        "scope": "ASAP 25-family counterfactual development; artificial complete erasure",
        "counts": {"families": len(effects), "interventions": len(rows)},
        "metrics": metrics, "gates": {key: bool(value) for key, value in gates.items()},
        "development_pass": bool(all(gates.values())),
        "kraisler_transfer_authorized": bool(all(gates.values())),
        "natural_warning_claim": False,
        "provenance": {"protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
                       "config_sha256": digest(CONFIG), "manifest_sha256": digest(source.MANIFEST)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("interventions.csv", rows), ("family_effects.csv", effects)):
        with (OUT / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

