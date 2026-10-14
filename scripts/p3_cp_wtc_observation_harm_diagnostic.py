#!/usr/bin/env python3
"""Post-failure diagnosis of CP-WTC observation harm; no confirmation gate."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_n_risk_balance as balance
import p3_tdr_cp_wtc_validation as cp
from score_following.temporal import healthy_before

CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_cp_wtc_observation_harm"


def diagnose(trace, cost, truth, score_hop, weights, start, stop):
    states = np.arange(trace.shape[1])
    corrections, ranks, gaps = [], [], []
    for frame in range(start, stop):
        distance = np.minimum(np.abs(states - truth[frame]) * score_hop, 5.0)
        prior = balance.predictive_prior(trace[frame - 1], weights)
        corrections.append(float(prior @ distance - trace[frame] @ distance))
        truth_state = int(np.clip(round(truth[frame]), 0, len(states) - 1))
        truth_cost = cost[frame, truth_state]
        ranks.append(float(np.mean(cost[frame] <= truth_cost)))
        gaps.append(float(truth_cost - cost[frame].min()))
    return {
        "observation_correction_sum": float(np.sum(corrections)),
        "negative_correction_share": float(np.mean(np.asarray(corrections) < 0)),
        "mean_truth_cost_rank": float(np.mean(ranks)),
        "mean_truth_cost_gap": float(np.mean(gaps)),
    }


def main():
    manifest = json.loads((ROOT / "results/p3_cp_wtc_preflight/manifest.json").read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")); hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows, healthy = [], 0
    for number, case in enumerate(row for row in manifest["rows"] if row["eligible"]):
        u, v, score_hop, truth, _ = cp.load_case(case, hop)
        cost = dose.cosine_cost(u, v); _, trace = dose.bayes_trace(cost, weights)
        error = np.abs(trace.argmax(axis=1) - truth) * score_hop
        start = int(round(15 / hop)); stop = start + int(round(10 / hop))
        is_healthy = stop < len(trace) and healthy_before(error, start - 1, 20, .5)
        healthy += int(is_healthy)
        row = {"artist": case["artist"], "piece": case["piece"], "healthy_at_15s": int(is_healthy),
               "pre_2s_mean_map_error": float(np.mean(error[start-20:start]))}
        if is_healthy:
            row.update(diagnose(trace, cost, truth, score_hop, weights, start, stop))
        else:
            row.update({"observation_correction_sum": "", "negative_correction_share": "",
                        "mean_truth_cost_rank": "", "mean_truth_cost_gap": ""})
        rows.append(row)
        print(f"[{number + 1}/50] {case['artist']} {case['piece']} healthy={is_healthy}")
    selected = [row for row in rows if row["healthy_at_15s"]]
    corrections = np.asarray([row["observation_correction_sum"] for row in selected])
    result = {
        "scope": "post-failure diagnosis only; cannot rescue P3-TDR",
        "counts": {"eligible_performances": len(rows), "healthy_performances": healthy,
                   "healthy_pieces": len({row["piece"] for row in selected})},
        "metrics": {
            "mean_10s_observation_correction": float(corrections.mean()),
            "negative_correction_performances": int(np.sum(corrections < 0)),
            "mean_negative_frame_share": float(np.mean([row["negative_correction_share"] for row in selected])),
            "mean_truth_cost_rank": float(np.mean([row["mean_truth_cost_rank"] for row in selected])),
            "mean_truth_cost_gap": float(np.mean([row["mean_truth_cost_gap"] for row in selected])),
        },
        "interpretation": "negative correction means the transcribed-MIDI observation increases GT-relative posterior risk",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "performances.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
