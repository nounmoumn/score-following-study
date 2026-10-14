#!/usr/bin/env python3
"""Frozen cross-corpus transfer of P3-PT risk and incidence endpoints."""

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
sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_i_natural_equivalence as natural
import p3_p_healthy_erasure as erasure
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_PT_TEMPORAL_RISK_PROTOCOL.md"
OUT = ROOT / "results/p3_pt_kraisler_transfer"
EXPOSURE = 20
CONFIRM = 4
FINAL = 5
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows):
    output = []
    for track in sorted({row["track"] for row in rows}):
        group = [row for row in rows if row["track"] == track]
        output.append({"track": track,
                       "risk_effect": float(np.mean([row["risk_effect"] for row in group])),
                       "incidence_effect": float(np.mean([row["erased_event"] - row["intact_event"]
                                                          for row in group]))})
    return output


def ci(values, rng):
    values = np.asarray(values)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    from p3_kraisler_signal_run import load_case
    from p3_batik_external_run import load_gt

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, duration = load_case(case, hop)
        beats, perf_times = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(u)) * hop, perf_times, beats * .5) / score_hop
        cost = dose.cosine_cost(u, v)
        _, trace = dose.bayes_trace(cost, weights)
        intact_error_full = np.abs(trace.argmax(axis=1) - truth) * score_hop
        for time_sec in np.arange(15., duration - 5. + 1e-9, 5.):
            end = int(round(time_sec / hop))
            stop = end + EXPOSURE + CONFIRM
            if stop >= len(cost) or not healthy_before(intact_error_full, end - 1, 20, .5):
                continue
            intact = natural.replay(trace[end - 1], cost[end:stop], weights)
            erased_cost = np.broadcast_to(cost[end:stop].mean(axis=1, keepdims=True), cost[end:stop].shape)
            erased = natural.replay(trace[end - 1], erased_cost, weights)
            local_truth = truth[end:stop]
            intact_risk = erasure.expected_risk(intact, local_truth, score_hop)
            erased_risk = erasure.expected_risk(erased, local_truth, score_hop)
            intact_error = np.abs(intact.argmax(axis=1) - local_truth) * score_hop
            erased_error = np.abs(erased.argmax(axis=1) - local_truth) * score_hop
            rows.append({
                "track": case["track"], "time_sec": float(time_sec),
                "risk_effect": float(np.mean(erased_risk[EXPOSURE-FINAL:EXPOSURE]
                                             - intact_risk[EXPOSURE-FINAL:EXPOSURE])),
                "intact_event": int(sustained_start(intact_error, 0, EXPOSURE) is not None),
                "erased_event": int(sustained_start(erased_error, 0, EXPOSURE) is not None),
                "intact_replay_residual": float(np.max(np.abs(intact - trace[end:stop]))),
                "erased_row_range": float(np.max(np.ptp(erased_cost, axis=1))),
            })
        print(f"[{number}/{len(cases)}] track={case['track']}")

    effects = aggregate(rows)
    risk = np.asarray([row["risk_effect"] for row in effects])
    incidence = np.asarray([row["incidence_effect"] for row in effects])
    rng = np.random.default_rng(SEED)
    metrics = {
        "mean_track_risk_effect": float(risk.mean()), "risk_effect_bootstrap_95": ci(risk, rng),
        "positive_track_risk_share": float(np.mean(risk > 0)),
        "mean_track_incidence_effect": float(incidence.mean()),
        "incidence_effect_bootstrap_95": ci(incidence, rng),
        "intact_event_rate": float(np.mean([row["intact_event"] for row in rows])),
        "erased_event_rate": float(np.mean([row["erased_event"] for row in rows])),
    }
    gates = {
        "coverage": len(effects) == 13 and len(rows) >= 100,
        "intact_replay_residual_le_1e_12": max(row["intact_replay_residual"] for row in rows) <= 1e-12,
        "erased_rows_flat": max(row["erased_row_range"] for row in rows) <= 1e-12,
        "risk_effect_ge_0_05": risk.mean() >= .05,
        "risk_ci_lower_gt_0": metrics["risk_effect_bootstrap_95"][0] > 0,
        "risk_positive_share_ge_0_70": np.mean(risk > 0) >= .70,
        "incidence_effect_ge_0_03": incidence.mean() >= .03,
        "incidence_ci_lower_gt_0": metrics["incidence_effect_bootstrap_95"][0] > 0,
    }
    result = {
        "hypothesis": "P3-PT healthy-state temporal risk/incidence",
        "scope": "KRAISLER cross-corpus transfer; complete artificial erasure",
        "counts": {"tracks": len(effects), "interventions": len(rows)},
        "metrics": metrics, "gates": {key: bool(value) for key, value in gates.items()},
        "transfer_pass": bool(all(gates.values())),
        "cross_corpus_p3_pt_confirmed": bool(all(gates.values())),
        "original_natural_p3_confirmed": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("interventions.csv", rows), ("track_effects.csv", effects)):
        with (OUT / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

