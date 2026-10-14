#!/usr/bin/env python3
"""Frozen ASAP development of P3-TD healthy-state duration response."""

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
import p3_i_natural_equivalence as natural
import p3_o_balance_transfer as source
import p3_p_healthy_erasure as erasure
from p3_short_signal import sustained_start

CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_TD_DURATION_RESPONSE_PROTOCOL.md"
OUT = ROOT / "results/p3_td_duration_response"
DURATIONS = (2, 5, 10)
LANDMARKS = tuple(range(15, 76, 10))
FINAL = 5
CONFIRM = 4
BOOTSTRAPS = 20000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows, unit_key):
    doses = []
    for unit in sorted({row[unit_key] for row in rows}):
        for duration in DURATIONS:
            group = [row for row in rows if row[unit_key] == unit and row["duration_sec"] == duration]
            doses.append({unit_key: unit, "duration_sec": duration,
                          "risk_effect": float(np.mean([row["risk_effect"] for row in group])),
                          "incidence_effect": float(np.mean([row["erased_event"] - row["intact_event"]
                                                             for row in group]))})
    effects = []
    for unit in sorted({row[unit_key] for row in doses}):
        group = sorted([row for row in doses if row[unit_key] == unit], key=lambda row: row["duration_sec"])
        effects.append({unit_key: unit,
                        "risk_duration_rho": float(spearmanr(DURATIONS,
                                                              [row["risk_effect"] for row in group]).statistic)})
    return doses, effects


def ci(values, rng):
    values = np.asarray(values)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def evaluate(rows, unit_key, coverage_gate):
    doses, effects = aggregate(rows, unit_key)
    curve = []
    for duration in DURATIONS:
        group = [row for row in doses if row["duration_sec"] == duration]
        curve.append({"duration_sec": duration,
                      "risk_effect": float(np.mean([row["risk_effect"] for row in group])),
                      "incidence_effect": float(np.mean([row["incidence_effect"] for row in group]))})
    ten = [row for row in doses if row["duration_sec"] == 10]
    ten_risk = np.asarray([row["risk_effect"] for row in ten])
    ten_incidence = np.asarray([row["incidence_effect"] for row in ten])
    rng = np.random.default_rng(SEED)
    risk_ci = ci(ten_risk, rng); incidence_ci = ci(ten_incidence, rng)
    risk_curve = [row["risk_effect"] for row in curve]
    incidence_curve = [row["incidence_effect"] for row in curve]
    gates = {
        "coverage": bool(coverage_gate(len(effects), len(rows) // len(DURATIONS))),
        "replay_and_manipulation": (max(row["intact_replay_residual"] for row in rows) <= 1e-12
                                    and max(row["erased_row_range"] for row in rows) <= 1e-12),
        "risk_curve_strictly_increasing": all(np.diff(risk_curve) > 0),
        "positive_risk_rho_share_ge_0_80": np.mean([row["risk_duration_rho"] > 0 for row in effects]) >= .80,
        "ten_sec_risk_ge_0_25": ten_risk.mean() >= .25,
        "ten_sec_risk_ci_lower_gt_0": risk_ci[0] > 0,
        "incidence_curve_nondecreasing": all(np.diff(incidence_curve) >= 0),
        "ten_sec_incidence_ge_0_10": ten_incidence.mean() >= .10,
        "ten_sec_incidence_ci_lower_gt_0": incidence_ci[0] > 0,
    }
    return doses, effects, curve, risk_ci, incidence_ci, gates


def main():
    cases = source.read_csv(source.MANIFEST)
    annotations = json.loads(source.ANNOTATIONS.read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")); hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth = source.load_case(case, annotations, hop)
        cost = dose.cosine_cost(u, v); _, trace = dose.bayes_trace(cost, weights)
        full_error = np.abs(trace.argmax(axis=1) - truth) * score_hop
        for time_sec in LANDMARKS:
            end = int(round(time_sec / hop)); stop = end + 100 + CONFIRM
            if stop >= len(cost) or np.max(full_error[end-20:end]) > .5:
                continue
            for duration in DURATIONS:
                frames = int(round(duration / hop)); branch_stop = end + frames + CONFIRM
                intact = natural.replay(trace[end-1], cost[end:branch_stop], weights)
                erased_cost = np.broadcast_to(cost[end:branch_stop].mean(axis=1, keepdims=True),
                                              cost[end:branch_stop].shape)
                erased = natural.replay(trace[end-1], erased_cost, weights)
                local_truth = truth[end:branch_stop]
                intact_risk = erasure.expected_risk(intact, local_truth, score_hop)
                erased_risk = erasure.expected_risk(erased, local_truth, score_hop)
                intact_error = np.abs(intact.argmax(axis=1) - local_truth) * score_hop
                erased_error = np.abs(erased.argmax(axis=1) - local_truth) * score_hop
                rows.append({"family": case["work_family"], "time_sec": time_sec,
                             "duration_sec": duration,
                             "risk_effect": float(np.mean(erased_risk[frames-FINAL:frames]
                                                          - intact_risk[frames-FINAL:frames])),
                             "intact_event": int(sustained_start(intact_error, 0, frames) is not None),
                             "erased_event": int(sustained_start(erased_error, 0, frames) is not None),
                             "intact_replay_residual": float(np.max(np.abs(intact-trace[end:branch_stop]))),
                             "erased_row_range": float(np.max(np.ptp(erased_cost, axis=1)))})
        print(f"[{number}/{len(cases)}] {case['work_family']}")
    doses, effects, curve, risk_ci, incidence_ci, gates = evaluate(
        rows, "family", lambda units, landmarks: units == 25 and landmarks >= 125)
    result = {"hypothesis": "P3-TD healthy-state erasure duration response",
              "scope": "ASAP 25-family development",
              "counts": {"families": len(effects), "landmarks": len(rows)//len(DURATIONS), "runs": len(rows)},
              "mean_curve": curve, "ten_sec_risk_bootstrap_95": risk_ci,
              "ten_sec_incidence_bootstrap_95": incidence_ci,
              "positive_risk_rho_share": float(np.mean([row["risk_duration_rho"] > 0 for row in effects])),
              "gates": {key: bool(value) for key, value in gates.items()},
              "development_pass": bool(all(gates.values())),
              "transfer_authorized": bool(all(gates.values())),
              "original_natural_p3_confirmed": False,
              "provenance": {"protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))}}
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("runs.csv", rows), ("unit_doses.csv", doses), ("unit_effects.csv", effects)):
        with (OUT/filename).open("w", encoding="utf8", newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()

