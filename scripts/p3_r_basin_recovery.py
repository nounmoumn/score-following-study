#!/usr/bin/env python3
"""Frozen ASAP development test of causal top-5 basin recovery for P3-R."""

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

OUT = ROOT / "results/p3_r_basin"
CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
BETA, RESTART, BASIN_SECONDS, K, SEED = 8.0, 1e-6, 2.0, 5, 20260920


def likelihood(cost_row):
    row = cost_row - float(np.min(cost_row))
    return np.exp(-BETA * row)


def get_centers(posterior, score_hop):
    radius = max(1, int(round(0.5 / score_hop)))
    separation = max(radius + 1, int(round(2.0 / score_hop)))
    return [center for center, _ in candidates(posterior, radius, separation, K)]


def branch_states(cost, posterior, centers, transition, score_hop, restore, decision):
    radius = max(1, int(round(BASIN_SECONDS / score_hop)))
    states, scores, n = [], [], posterior.size
    for center in centers:
        lo, hi = max(0, center - radius), min(n, center + radius + 1)
        state = np.zeros_like(posterior)
        mass = float(np.sum(posterior[lo:hi]))
        if mass <= 0.0:
            raise RuntimeError("candidate basin has zero posterior mass")
        state[lo:hi] = posterior[lo:hi] / mass
        states.append(state)
        scores.append(math.log(mass))
    for frame in range(restore, decision):
        emit = likelihood(cost[frame])
        for j, state in enumerate(states):
            prior = predict_prior(state, transition, 0.0)
            weighted = prior * emit
            evidence = float(np.sum(weighted))
            if evidence <= 0.0 or not np.isfinite(evidence):
                raise RuntimeError("non-finite branch evidence")
            states[j] = weighted / evidence
            scores[j] += math.log(evidence)
    return states, np.asarray(scores, dtype=float)


def continue_path(cost, state, transition, decision, end):
    out = np.empty(end - decision, dtype=int)
    for offset, frame in enumerate(range(decision, end)):
        prior = predict_prior(state, transition, RESTART)
        weighted = prior * likelihood(cost[frame])
        state = weighted / float(np.sum(weighted))
        out[offset] = int(np.argmax(state))
    return out


def burden(path, intact, score_hop, decision, eval_end):
    hi = min(eval_end, path.size, intact.size)
    return float(np.mean(np.abs(path[decision:hi] - intact[decision:hi])) * score_hop)


def work_bootstrap(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["composition"], []).append(row[key])
    work_values = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(work_values, size=(20_000, len(work_values)), replace=True).mean(axis=1)
    return {"mean": float(work_values.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    temporal = yaml.safe_load(TEMPORAL.read_text(encoding="utf-8"))
    hop = float(cfg["hop_sec"])
    onset = int(round(cfg["onset_sec"] / hop))
    restore = int(round(cfg["restore_sec"] / hop))
    decision = restore + int(round(2.0 / hop))
    eval_end = restore + int(round(10.0 / hop))
    end = int(round(cfg["duration_sec"] / hop))
    transition = np.asarray(temporal["bayes"]["transition"], dtype=float)
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
            u = notes_to_chroma_stream(pnotes, cfg["duration_sec"], hop)
            v = notes_to_chroma_stream(snotes, sd, score_hop)
            base_cost = cosine_cost(u, v)
            erased_cost = base_cost.copy()
            erased_cost[onset:restore] = erased_cost[onset:restore].mean(axis=1, keepdims=True)

            intact = bayes_from_cost(base_cost, transition, BETA, RESTART)
            natural = bayes_from_cost(erased_cost, transition, BETA, RESTART)
            erased_post, erased_prefix = posterior_until(erased_cost, restore, transition, BETA, RESTART)
            if not np.array_equal(erased_prefix, natural[:restore]):
                raise AssertionError("erased prefix mismatch")
            erased_centers = get_centers(erased_post, score_hop)
            erased_states, erased_scores = branch_states(
                erased_cost, erased_post, erased_centers, transition, score_hop, restore, decision)
            selected_idx = int(np.argmax(erased_scores))
            correct_index = int(intact[restore - 1])
            oracle_idx = int(np.argmin(np.abs(np.asarray(erased_centers) - correct_index)))

            selected = natural.copy()
            selected[decision:end] = continue_path(
                erased_cost, erased_states[selected_idx].copy(), transition, decision, end)
            oracle = natural.copy()
            oracle[decision:end] = continue_path(
                erased_cost, erased_states[oracle_idx].copy(), transition, decision, end)

            intact_post, intact_prefix = posterior_until(base_cost, restore, transition, BETA, RESTART)
            if not np.array_equal(intact_prefix, intact[:restore]):
                raise AssertionError("intact prefix mismatch")
            intact_centers = get_centers(intact_post, score_hop)
            intact_states, intact_scores = branch_states(
                base_cost, intact_post, intact_centers, transition, score_hop, restore, decision)
            safety_idx = int(np.argmax(intact_scores))
            safety = intact.copy()
            safety[decision:end] = continue_path(
                base_cost, intact_states[safety_idx].copy(), transition, decision, end)

            natural_burden = burden(natural, intact, score_hop, decision, eval_end)
            selector_burden = burden(selected, intact, score_hop, decision, eval_end)
            oracle_burden = burden(oracle, intact, score_hop, decision, eval_end)
            safety_burden = burden(safety, intact, score_hop, decision, eval_end)
            rows.append({
                "composition": item["composition"], "performance": item["performance"],
                "score_hop_sec": score_hop, "natural_burden": natural_burden,
                "selector_burden": selector_burden, "oracle_burden": oracle_burden,
                "safety_burden": safety_burden,
                "selector_improvement": natural_burden - selector_burden,
                "oracle_improvement": natural_burden - oracle_burden,
                "selector_center_distance_sec": abs(erased_centers[selected_idx] - correct_index) * score_hop,
                "oracle_center_distance_sec": abs(erased_centers[oracle_idx] - correct_index) * score_hop,
                "selector_index": selected_idx, "oracle_index": oracle_idx,
                "selector_log_score_margin": float(np.sort(erased_scores)[-1] - np.sort(erased_scores)[-2]),
            })
            print(f"[{number}/{len(manifest)}] {item['performance']}")

    names = ("natural_burden", "selector_burden", "oracle_burden",
             "selector_improvement", "oracle_improvement", "safety_burden")
    metrics = {key: work_bootstrap(rows, key) for key in names}
    natural_mean = metrics["natural_burden"]["mean"]
    selector_imp, oracle_imp = metrics["selector_improvement"], metrics["oracle_improvement"]
    oracle_gate = oracle_imp["mean"] >= .20 * natural_mean and oracle_imp["ci95"][0] > 0
    selector_gate = selector_imp["mean"] >= .20 * natural_mean and selector_imp["ci95"][0] > 0
    fraction_gate = oracle_imp["mean"] > 0 and selector_imp["mean"] >= .50 * oracle_imp["mean"]
    safety_violation_rate = float(np.mean([r["safety_burden"] > .50 for r in rows]))
    safety_gate = metrics["safety_burden"]["mean"] <= .10 and safety_violation_rate <= .05
    gate_pass = bool(oracle_gate and selector_gate and fraction_gate and safety_gate)
    if not oracle_gate:
        decision_text = "branch_resolution_or_initialization_bottleneck"
    elif not selector_gate or not fraction_gate:
        decision_text = "branch_ranking_bottleneck"
    elif not safety_gate:
        decision_text = "intact_safety_failure"
    else:
        decision_text = "development_gate_pass_independent_validation_allowed"
    result = {
        "scope": "ASAP development basin recovery; not independent P3 confirmation",
        "protocol": {"n_performances": len(rows),
                     "n_works": len(set(r["composition"] for r in rows)),
                     "erase_frames": [onset, restore], "decision_frame": decision,
                     "evaluation_frames": [decision, eval_end], "candidate_k": K,
                     "basin_seconds": BASIN_SECONDS, "beta": BETA, "restart": RESTART,
                     "branch_restart": 0.0, "bootstrap_seed": SEED},
        "metrics": metrics,
        "diagnostics": {
            "selector_correct_basin_rate": float(np.mean([r["selector_center_distance_sec"] <= BASIN_SECONDS for r in rows])),
            "oracle_correct_basin_rate": float(np.mean([r["oracle_center_distance_sec"] <= BASIN_SECONDS for r in rows])),
            "safety_violation_rate": safety_violation_rate},
        "gates": {"oracle": bool(oracle_gate), "selector": bool(selector_gate),
                  "fraction_of_oracle": bool(fraction_gate), "intact_safety": bool(safety_gate),
                  "all": gate_pass},
        "decision": decision_text, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
