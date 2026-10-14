#!/usr/bin/env python3
"""Bayes-factor safeguarded P3-R recovery on frozen multisite interventions."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost, bayes_from_cost
from p3_r_candidate_audit import candidates, posterior_until
from p3_r_basin_recovery import continue_path, burden, work_bootstrap, likelihood

CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_r_safeguarded"
BETA, RESTART, K, BASIN_SECONDS = 8.0, 1e-6, 10, 2.0
LOG_BF = math.log(100.0)
WINDOWS_SEC = ((10, 20), (20, 30), (30, 40), (40, 50))


def get_centers(posterior, score_hop):
    radius = max(1, int(round(0.5 / score_hop)))
    separation = max(radius + 1, int(round(2.0 / score_hop)))
    return [center for center, _ in candidates(posterior, radius, separation, K)]


def compare(cost, posterior, centers, transition, score_hop, restore, decision):
    radius, n = max(1, int(round(BASIN_SECONDS / score_hop))), posterior.size
    states = []
    for center in centers:
        lo, hi = max(0, center - radius), min(n, center + radius + 1)
        state = np.zeros_like(posterior)
        mass = float(posterior[lo:hi].sum())
        state[lo:hi] = posterior[lo:hi] / mass
        states.append(state)
    branch_scores = np.zeros(len(states))
    null_state, null_score = posterior.copy(), 0.0
    for frame in range(restore, decision):
        emit = likelihood(cost[frame])
        null_prior = predict_prior(null_state, transition, 0.0)
        null_weighted = null_prior * emit
        znull = float(null_weighted.sum())
        null_state = null_weighted / znull
        null_score += math.log(znull)
        for j, state in enumerate(states):
            prior = predict_prior(state, transition, 0.0)
            weighted = prior * emit
            z = float(weighted.sum())
            states[j] = weighted / z
            branch_scores[j] += math.log(z)
    return states, branch_scores, null_score


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    temporal = yaml.safe_load(TEMPORAL.read_text(encoding="utf-8"))
    hop = float(cfg["hop_sec"])
    transition = np.asarray(temporal["bayes"]["transition"], dtype=float)
    end = int(round(cfg["duration_sec"] / hop))
    with (ROOT / cfg["manifest"]).open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    asap_root = ROOT / cfg["asap_root"]
    with (asap_root / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {row["midi_performance"]: row for row in csv.DictReader(handle)}

    rows = []
    with threadpool_limits(limits=1):
        for number, item in enumerate(manifest, 1):
            meta = metadata[item["performance"]]
            pnotes = load_notes(asap_root / meta["midi_performance"])
            snotes = load_notes(asap_root / meta["midi_score"])
            pd, sd = max(n[1] for n in pnotes), max(n[1] for n in snotes)
            score_hop = hop * sd / pd
            base = cosine_cost(notes_to_chroma_stream(pnotes, cfg["duration_sec"], hop),
                               notes_to_chroma_stream(snotes, sd, score_hop))
            intact = bayes_from_cost(base, transition, BETA, RESTART)
            for start_sec, restore_sec in WINDOWS_SEC:
                onset, restore = int(start_sec / hop), int(restore_sec / hop)
                decision, eval_end = restore + int(2 / hop), restore + int(10 / hop)
                erased = base.copy()
                erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
                natural = bayes_from_cost(erased, transition, BETA, RESTART)
                natural_burden = burden(natural, intact, score_hop, decision, eval_end)
                posterior, prefix = posterior_until(erased, restore, transition, BETA, RESTART)
                if not np.array_equal(prefix, natural[:restore]):
                    raise AssertionError("erased prefix mismatch")
                centers = get_centers(posterior, score_hop)
                states, scores, null_score = compare(
                    erased, posterior, centers, transition, score_hop, restore, decision)
                branch_burdens = []
                for state in states:
                    candidate_path = natural.copy()
                    candidate_path[decision:end] = continue_path(
                        erased, state.copy(), transition, decision, end)
                    branch_burdens.append(burden(candidate_path, intact, score_hop, decision, eval_end))
                selected_index = int(np.argmax(scores))
                log_bf = float(scores[selected_index] - null_score)
                switched = log_bf >= LOG_BF
                selector_burden = branch_burdens[selected_index] if switched else natural_burden
                oracle_burden = min([natural_burden] + branch_burdens)

                iposterior, iprefix = posterior_until(base, restore, transition, BETA, RESTART)
                if not np.array_equal(iprefix, intact[:restore]):
                    raise AssertionError("intact prefix mismatch")
                icenters = get_centers(iposterior, score_hop)
                istates, iscores, inull_score = compare(
                    base, iposterior, icenters, transition, score_hop, restore, decision)
                isel = int(np.argmax(iscores))
                intact_log_bf = float(iscores[isel] - inull_score)
                safety_switched = intact_log_bf >= LOG_BF
                if safety_switched:
                    safety_path = intact.copy()
                    safety_path[decision:end] = continue_path(
                        base, istates[isel].copy(), transition, decision, end)
                    safety_burden = burden(safety_path, intact, score_hop, decision, eval_end)
                else:
                    safety_burden = 0.0
                rows.append({
                    "composition": item["composition"], "performance": item["performance"],
                    "window_sec": [start_sec, restore_sec], "natural_burden": natural_burden,
                    "selector_burden": selector_burden, "oracle_burden": oracle_burden,
                    "safety_burden": safety_burden,
                    "selector_improvement": natural_burden - selector_burden,
                    "oracle_improvement": natural_burden - oracle_burden,
                    "selector_log_bf": log_bf, "selector_switched": switched,
                    "safety_log_bf": intact_log_bf, "safety_switched": safety_switched})
            print(f"[{number}/{len(manifest)}] {item['performance']}")

    names = ("natural_burden", "selector_burden", "oracle_burden",
             "selector_improvement", "oracle_improvement", "safety_burden")
    metrics = {key: work_bootstrap(rows, key) for key in names}
    affected = [row for row in rows if row["natural_burden"] > .5]
    event_gate = len(affected) >= 20 and len(set(r["composition"] for r in affected)) >= 10
    natural_mean, oi, si = (metrics["natural_burden"]["mean"],
                            metrics["oracle_improvement"], metrics["selector_improvement"])
    oracle_gate = oi["mean"] >= .2 * natural_mean and oi["ci95"][0] > 0
    selector_gate = si["mean"] >= .2 * natural_mean and si["ci95"][0] > 0
    fraction_gate = oi["mean"] > 0 and si["mean"] >= .5 * oi["mean"]
    violation_rate = float(np.mean([r["safety_burden"] > .5 for r in rows]))
    safety_gate = metrics["safety_burden"]["mean"] <= .1 and violation_rate <= .05
    all_gate = bool(event_gate and oracle_gate and selector_gate and fraction_gate and safety_gate)
    if not oracle_gate: decision_text = "candidate_set_headroom_not_supported"
    elif not safety_gate: decision_text = "bayes_factor_safety_failure"
    elif not selector_gate or not fraction_gate: decision_text = "gt_free_ranking_bottleneck"
    else: decision_text = "development_gate_pass_independent_validation_allowed"
    result = {
        "scope": "ASAP safeguarded multisite development; not independent confirmation",
        "protocol": {"interventions": len(rows), "candidate_k": K,
                     "log_bayes_factor_threshold": LOG_BF, "windows_sec": WINDOWS_SEC},
        "event_incidence": {"persistent_events": len(affected),
                            "affected_works": len(set(r["composition"] for r in affected))},
        "metrics": metrics,
        "switching": {"selector_count": sum(r["selector_switched"] for r in rows),
                      "selector_affected_count": sum(r["selector_switched"] for r in affected),
                      "safety_count": sum(r["safety_switched"] for r in rows),
                      "safety_violation_rate": violation_rate},
        "gates": {"event_incidence": event_gate, "oracle": bool(oracle_gate),
                  "selector": bool(selector_gate), "fraction_of_oracle": bool(fraction_gate),
                  "intact_safety": bool(safety_gate), "all": all_gate},
        "decision": decision_text, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
