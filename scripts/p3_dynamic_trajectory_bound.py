#!/usr/bin/env python3
"""Rigorous gridded upper bound for continuous correction trajectories."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_observation_loss_selector as selector
import p3_vienna_forecast_validation as vienna
from p3_fhb_correction_burst_transfer import fast_auc

OUT = ROOT / "results/p3_dynamic_trajectory_bound"
RADIUS = 0.5211881394162992
HISTORY_SEC = 2.0
CAP = 5.0
GRID_STEP = 0.25
MAX_STEP = 3.0


def correction_at(delta, y, score_hop):
    states = np.arange(len(delta), dtype=float)
    return -float(np.minimum(np.abs(states - y) * score_hop, CAP) @ delta)


def dynamic_upper_bound(trace, end, score_hop, weights, hop):
    history = int(round(HISTORY_SEC / hop)); half = GRID_STEP / 2
    dp = None; error = 0.0
    for frame in range(end - history, end):
        prior = dc.apply_prediction(trace[frame - 1], weights)
        delta = trace[frame] - prior
        center = selector.median_state(trace[frame])
        lo = max(0.0, center - RADIUS / score_hop)
        hi = min(len(delta) - 1.0, center + RADIUS / score_hop)
        grid_lo = np.floor((lo - half) / GRID_STEP) * GRID_STEP
        grid_hi = np.ceil((hi + half) / GRID_STEP) * GRID_STEP
        grid = np.arange(grid_lo, grid_hi + GRID_STEP / 2, GRID_STEP)
        values = np.asarray([correction_at(delta, y, score_hop) for y in grid])
        error += score_hop * float(np.abs(delta).sum()) * half
        if dp is None:
            dp = (grid, values)
            continue
        previous_grid, previous_value = dp
        current = np.full(len(grid), -np.inf)
        for index, y in enumerate(grid):
            feasible = (y - previous_grid >= -GRID_STEP - 1e-12) & (y - previous_grid <= MAX_STEP + GRID_STEP + 1e-12)
            if np.any(feasible):
                current[index] = values[index] + float(np.max(previous_value[feasible]))
        dp = (grid, current)
    upper = float(np.max(dp[1])) + error
    return {"trajectory_upper": upper, "trajectory_score": -upper, "lipschitz_error": error}


def assumption_holds(trace, truth, end, score_hop, hop):
    history = int(round(HISTORY_SEC / hop)); start = end - history
    centers = np.asarray([selector.median_state(trace[frame]) for frame in range(start, end)])
    actual = np.asarray(truth[start:end])
    interval = bool(np.all(np.abs(actual - centers) * score_hop <= RADIUS + 1e-12))
    increments = np.diff(actual)
    speed = bool(np.all(increments >= -1e-12) and np.all(increments <= MAX_STEP + 1e-12))
    return interval and speed, interval, speed


def summarize(rows):
    positive = [row["trajectory_score"] for row in rows if row["observation_loss"]]
    negative = [row["trajectory_score"] for row in rows if not row["observation_loss"]]
    return {"events": len(rows), "observation_loss": len(positive),
            "assumption_coverage": float(np.mean([row["assumption_holds"] for row in rows])),
            "interval_coverage": float(np.mean([row["interval_holds"] for row in rows])),
            "speed_coverage": float(np.mean([row["speed_holds"] for row in rows])),
            "positive_certificates": int(sum(row["trajectory_score"] > 0 for row in rows)),
            "observation_loss_median": float(np.median(positive)), "other_median": float(np.median(negative)),
            "auc": fast_auc(positive, negative)}


def main():
    cfg = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text()); hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"]); rows = []
    from p3_kraisler_signal_run import load_case
    from p3_batik_external_run import load_gt
    k_events = list(csv.DictReader((ROOT / "results/p3_observation_loss_selector/events.csv").open(encoding="utf8")))
    manifest = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text())
    for case in (x for x in manifest["rows"] if x["eligible"]):
        selected = [x for x in k_events if x["track"] == case["track"]]
        if not selected: continue
        obs, ref, score_hop, _ = load_case(case, hop); clock, perf = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(obs)) * hop, perf, clock * 0.5) / score_hop
        _, trace = dose.bayes_trace(dose.cosine_cost(obs, ref), weights)
        for event in selected:
            end = int(round(float(event["landmark_sec"]) / hop)); bound = dynamic_upper_bound(trace, end, score_hop, weights, hop)
            holds, interval, speed = assumption_holds(trace, truth, end, score_hop, hop)
            rows.append({"corpus":"KRAISLER","track":case["track"],"observation_loss":event["observation_loss"]=="True",
                         "assumption_holds":holds,"interval_holds":interval,"speed_holds":speed,**bound})
        print(f"[KRAISLER] {case['track']}", flush=True)
    v_events = list(csv.DictReader((ROOT / "results/p3_vienna_observation_shock_validation/events.csv").open(encoding="utf8")))
    vm = json.loads((ROOT / "results/p3_vienna_forecast_preflight/manifest.json").read_text()); cache={}
    for case in (x for x in vm["rows"] if x["split"]=="validation"):
        selected=[x for x in v_events if x["track"]==case["stem"]]
        if not selected: continue
        obs,ref,score_hop,truth,_=vienna.load_case(case,hop,cache);_,trace=dose.bayes_trace(dose.cosine_cost(obs,ref),weights)
        for event in selected:
            end=int(round(float(event["time_sec"])/hop));bound=dynamic_upper_bound(trace,end,score_hop,weights,hop)
            holds,interval,speed=assumption_holds(trace,truth,end,score_hop,hop)
            rows.append({"corpus":"Vienna","track":case["stem"],"observation_loss":event["observation_loss"]=="True",
                         "assumption_holds":holds,"interval_holds":interval,"speed_holds":speed,**bound})
        print(f"[Vienna] {case['stem']}",flush=True)
    corpora={name:summarize([r for r in rows if r["corpus"]==name]) for name in ("KRAISLER","Vienna")};pooled=summarize(rows)
    gates={"assumption_coverage_ge_0_90_each":all(x["assumption_coverage"]>=.90 for x in corpora.values()),
           "auc_ge_0_65_each":all(x["auc"]>=.65 for x in corpora.values()),"pooled_auc_ge_0_70":pooled["auc"]>=.70}
    result={"scope":"final opened-corpus selector development route","bound":"continuous trajectory upper bound via expanded grid plus Lipschitz error",
            "corpora":corpora,"pooled":pooled,"gates":gates,"qualified_for_future_independent_freeze":bool(all(gates.values())),
            "existing_cohort_selector_development_complete":True,"natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"events.csv").open("w",newline="",encoding="utf8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
