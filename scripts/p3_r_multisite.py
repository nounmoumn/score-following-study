#!/usr/bin/env python3
"""Frozen multisite ASAP development experiment for P3-R."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from p3_r_candidate_audit import candidates, posterior_until
from p3_r_basin_recovery import branch_states, continue_path, burden, work_bootstrap

CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_r_multisite"
BETA, RESTART, K = 8.0, 1e-6, 10
WINDOWS_SEC = ((10, 20), (20, 30), (30, 40), (40, 50))


def centers10(posterior, score_hop):
    radius = max(1, int(round(0.5 / score_hop)))
    separation = max(radius + 1, int(round(2.0 / score_hop)))
    return [center for center, _ in candidates(posterior, radius, separation, K)]


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
            u = notes_to_chroma_stream(pnotes, cfg["duration_sec"], hop)
            v = notes_to_chroma_stream(snotes, sd, score_hop)
            base = cosine_cost(u, v)
            intact = bayes_from_cost(base, transition, BETA, RESTART)

            for start_sec, restore_sec in WINDOWS_SEC:
                onset, restore = int(start_sec / hop), int(restore_sec / hop)
                decision, eval_end = restore + int(2 / hop), restore + int(10 / hop)
                erased = base.copy()
                erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
                natural = bayes_from_cost(erased, transition, BETA, RESTART)
                erased_post, prefix = posterior_until(erased, restore, transition, BETA, RESTART)
                if not np.array_equal(prefix, natural[:restore]):
                    raise AssertionError("erased prefix mismatch")
                ecenters = centers10(erased_post, score_hop)
                estates, escores = branch_states(
                    erased, erased_post, ecenters, transition, score_hop, restore, decision)
                selected_index = int(np.argmax(escores))
                correct = int(intact[restore - 1])
                oracle_index = int(np.argmin(np.abs(np.asarray(ecenters) - correct)))
                selected, oracle = natural.copy(), natural.copy()
                selected[decision:end] = continue_path(
                    erased, estates[selected_index].copy(), transition, decision, end)
                oracle[decision:end] = continue_path(
                    erased, estates[oracle_index].copy(), transition, decision, end)

                intact_post, intact_prefix = posterior_until(base, restore, transition, BETA, RESTART)
                if not np.array_equal(intact_prefix, intact[:restore]):
                    raise AssertionError("intact prefix mismatch")
                icenters = centers10(intact_post, score_hop)
                istates, iscores = branch_states(
                    base, intact_post, icenters, transition, score_hop, restore, decision)
                safety_index = int(np.argmax(iscores))
                safety = intact.copy()
                safety[decision:end] = continue_path(
                    base, istates[safety_index].copy(), transition, decision, end)

                nb = burden(natural, intact, score_hop, decision, eval_end)
                sb = burden(selected, intact, score_hop, decision, eval_end)
                ob = burden(oracle, intact, score_hop, decision, eval_end)
                ib = burden(safety, intact, score_hop, decision, eval_end)
                rows.append({
                    "composition": item["composition"], "performance": item["performance"],
                    "window_sec": [start_sec, restore_sec], "natural_burden": nb,
                    "selector_burden": sb, "oracle_burden": ob, "safety_burden": ib,
                    "selector_improvement": nb - sb, "oracle_improvement": nb - ob,
                    "selector_center_distance_sec": abs(ecenters[selected_index] - correct) * score_hop,
                    "oracle_center_distance_sec": abs(ecenters[oracle_index] - correct) * score_hop,
                })
            print(f"[{number}/{len(manifest)}] {item['performance']}")

    names = ("natural_burden", "selector_burden", "oracle_burden",
             "selector_improvement", "oracle_improvement", "safety_burden")
    metrics = {key: work_bootstrap(rows, key) for key in names}
    affected = [row for row in rows if row["natural_burden"] > 0.5]
    affected_works = len(set(row["composition"] for row in affected))
    event_gate = len(affected) >= 20 and affected_works >= 10
    natural_mean = metrics["natural_burden"]["mean"]
    oi, si = metrics["oracle_improvement"], metrics["selector_improvement"]
    oracle_gate = oi["mean"] >= .2 * natural_mean and oi["ci95"][0] > 0
    selector_gate = si["mean"] >= .2 * natural_mean and si["ci95"][0] > 0
    fraction_gate = oi["mean"] > 0 and si["mean"] >= .5 * oi["mean"]
    safety_violation_rate = float(np.mean([row["safety_burden"] > .5 for row in rows]))
    safety_gate = metrics["safety_burden"]["mean"] <= .1 and safety_violation_rate <= .05
    all_gate = bool(event_gate and oracle_gate and selector_gate and fraction_gate and safety_gate)
    if not event_gate:
        decision_text = "insufficient_persistent_event_incidence"
    elif not oracle_gate:
        decision_text = "candidate_initialization_bottleneck"
    elif not selector_gate or not fraction_gate:
        decision_text = "gt_free_ranking_bottleneck"
    elif not safety_gate:
        decision_text = "intact_safety_failure"
    else:
        decision_text = "development_gate_pass_independent_validation_allowed"
    result = {
        "scope": "ASAP multisite development; not independent P3 confirmation",
        "protocol": {"performances": len(manifest), "works": len(set(r["composition"] for r in rows)),
                     "windows_sec": WINDOWS_SEC, "interventions": len(rows), "candidate_k": K,
                     "basin_seconds": 2.0, "decision_delay_sec": 2.0,
                     "evaluation_duration_sec": 8.0},
        "event_incidence": {"persistent_events": len(affected),
                            "persistent_event_rate": len(affected) / len(rows),
                            "affected_works": affected_works},
        "metrics": metrics,
        "affected_diagnostics": {
            "natural_burden_mean": float(np.mean([r["natural_burden"] for r in affected])) if affected else 0,
            "selector_improvement_mean": float(np.mean([r["selector_improvement"] for r in affected])) if affected else 0,
            "oracle_improvement_mean": float(np.mean([r["oracle_improvement"] for r in affected])) if affected else 0},
        "safety_violation_rate": safety_violation_rate,
        "gates": {"event_incidence": event_gate, "oracle": bool(oracle_gate),
                  "selector": bool(selector_gate), "fraction_of_oracle": bool(fraction_gate),
                  "intact_safety": bool(safety_gate), "all": all_gate},
        "decision": decision_text, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
