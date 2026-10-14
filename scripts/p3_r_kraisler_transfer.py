#!/usr/bin/env python3
"""One-shot frozen KRAISLER transfer of the complete P3-R pipeline."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from score_following.p3_extension import cosine_cost, bayes_from_cost
from p3_kraisler_signal_run import load_case
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden, continue_path
from p3_r_trigger_audit import (
    FEATURES, centers_and_masses, compare_features, posterior_stats,
)

CONFIG = ROOT / "configs/p3_temporal.yaml"
MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
MODEL_RESULT = ROOT / "results/p3_r_trigger/result.json"
OUT = ROOT / "results/p3_r_kraisler_transfer"
WINDOWS_SEC = ((10, 20), (20, 30), (30, 40), (40, 50))
BETA, RESTART, SEED = 8.0, 1e-6, 20260920


def frozen_score(features, model):
    x = np.asarray([features[name] for name in model["feature_order"]])
    z = (x - np.asarray(model["scaler_mean"])) / np.asarray(model["scaler_scale"])
    logit = float(np.asarray(model["coefficient"]) @ z + model["intercept"])
    return 1.0 / (1.0 + math.exp(-np.clip(logit, -700, 700)))


def extract(cost, posterior, path, transition, score_hop, onset, restore, decision):
    centers, masses = centers_and_masses(posterior, score_hop)
    states, null, scores, endpoints, selected, null_score = compare_features(
        cost, posterior, centers, transition, score_hop, restore, decision)
    re, rm, rs = posterior_stats(posterior, score_hop)
    de, dm, ds = posterior_stats(null, score_hop)
    segment = path[onset:restore]
    nonlocal_radius = max(len(transition)-1, int(round(.5/score_hop)))
    mass_prob = masses / masses.sum()
    mass_entropy = -float(np.sum(mass_prob*np.log(np.maximum(mass_prob, 1e-300)))) / math.log(len(mass_prob))
    ordered = np.sort(scores)
    features = {
        "restore_entropy": re, "restore_max_probability": rm, "restore_std_seconds": rs,
        "decision_entropy": de, "decision_max_probability": dm, "decision_std_seconds": ds,
        "erased_net_progress_seconds": float((segment[-1]-segment[0])*score_hop),
        "erased_unique_fraction": float(len(np.unique(segment))/len(segment)),
        "erased_nonlocal_switches": int(np.sum(np.abs(np.diff(segment)) > nonlocal_radius)),
        "candidate_max_mass": float(masses.max()), "candidate_mass_entropy": mass_entropy,
        "max_log_bf": float(scores[selected]-null_score),
        "score_margin": float(ordered[-1]-ordered[-2]),
        "selected_null_distance_seconds": float(abs(endpoints[selected]-np.argmax(null))*score_hop),
        "candidate_endpoint_spread_seconds": float((endpoints.max()-endpoints.min())*score_hop),
    }
    if tuple(features) != FEATURES:
        raise AssertionError("feature order drift")
    return features, states, selected


def track_bootstrap(rows, key):
    grouped = {}
    for row in rows: grouped.setdefault(row["track"], []).append(row[key])
    values = np.asarray([np.mean(x) for x in grouped.values()])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(draws, [.025, .975])]}


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); hop = float(cfg["hop_sec"])
    transition = np.asarray(cfg["bayes"]["transition"], dtype=float)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = [case for case in manifest["rows"] if case["eligible"]]
    development = json.loads(MODEL_RESULT.read_text(encoding="utf-8"))
    if not development["gate_pass"]: raise AssertionError("development trigger gate did not pass")
    model = development["frozen_full_development_model"]
    rows = []
    with threadpool_limits(limits=1):
        for number, case in enumerate(cases, 1):
            u, v, score_hop, duration = load_case(case, hop)
            base = cosine_cost(u, v); intact = bayes_from_cost(base, transition, BETA, RESTART)
            for start_sec, restore_sec in WINDOWS_SEC:
                onset, restore = int(start_sec/hop), int(restore_sec/hop)
                decision, eval_end = restore+int(2/hop), restore+int(10/hop)
                if eval_end > len(base): continue
                erased = base.copy(); erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
                natural = bayes_from_cost(erased, transition, BETA, RESTART)
                natural_burden = burden(natural, intact, score_hop, decision, eval_end)
                posterior, prefix = posterior_until(erased, restore, transition, BETA, RESTART)
                if not np.array_equal(prefix, natural[:restore]): raise AssertionError("erased prefix mismatch")
                features, states, selected = extract(erased, posterior, prefix, transition, score_hop, onset, restore, decision)
                probability = frozen_score(features, model); trigger = probability >= model["threshold"]
                branch_burdens = []
                for state in states:
                    path = natural.copy(); path[decision:] = continue_path(erased, state.copy(), transition, decision, len(base))
                    branch_burdens.append(burden(path, intact, score_hop, decision, eval_end))
                selector_burden = branch_burdens[selected] if trigger else natural_burden
                oracle_burden = min([natural_burden] + branch_burdens)

                iposterior, iprefix = posterior_until(base, restore, transition, BETA, RESTART)
                if not np.array_equal(iprefix, intact[:restore]): raise AssertionError("intact prefix mismatch")
                ifeatures, istates, iselected = extract(base, iposterior, iprefix, transition, score_hop, onset, restore, decision)
                iprobability = frozen_score(ifeatures, model); safety_trigger = iprobability >= model["threshold"]
                if safety_trigger:
                    safety_path = intact.copy(); safety_path[decision:] = continue_path(base, istates[iselected].copy(), transition, decision, len(base))
                    safety_burden = burden(safety_path, intact, score_hop, decision, eval_end)
                else: safety_burden = 0.
                rows.append({"track": case["track"], "composer": case["composer"], "window_sec": [start_sec, restore_sec],
                             "persistent_event": natural_burden > .5, "trigger_probability": probability,
                             "trigger": trigger, "natural_burden": natural_burden,
                             "selector_burden": selector_burden, "oracle_burden": oracle_burden,
                             "selector_improvement": natural_burden-selector_burden,
                             "oracle_improvement": natural_burden-oracle_burden,
                             "safety_probability": iprobability, "safety_trigger": safety_trigger,
                             "safety_burden": safety_burden})
            print(f"[{number}/{len(cases)}] track {case['track']}")

    y = np.asarray([r["persistent_event"] for r in rows]); score = np.asarray([r["trigger_probability"] for r in rows])
    pred = np.asarray([r["trigger"] for r in rows]); prevalence = float(y.mean())
    auc = float(roc_auc_score(y, score)) if y.any() and (~y).any() else None
    ap = float(average_precision_score(y, score)) if y.any() else None
    recall = float(np.sum(pred & y)/np.sum(y)) if y.any() else 0.
    fpr = float(np.sum(pred & ~y)/np.sum(~y)) if (~y).any() else 0.
    affected_tracks = len(set(r["track"] for r in rows if r["persistent_event"]))
    names = ("natural_burden","selector_burden","oracle_burden","selector_improvement","oracle_improvement","safety_burden")
    metrics = {name: track_bootstrap(rows,name) for name in names}
    coverage = len(set(r["track"] for r in rows)) >= 10 and len(rows) >= 40
    incidence = int(y.sum()) >= 8 and affected_tracks >= 5
    trigger_gate = auc is not None and auc >= .70 and ap >= 2*prevalence and recall >= .35 and fpr <= .10
    natural_mean = metrics["natural_burden"]["mean"]; oi=metrics["oracle_improvement"]; si=metrics["selector_improvement"]
    oracle_gate = oi["mean"] >= .2*natural_mean and oi["ci95"][0] > 0
    selector_gate = si["mean"] >= .2*natural_mean and si["ci95"][0] > 0
    fraction_gate = oi["mean"] > 0 and si["mean"] >= .5*oi["mean"]
    safety_rate = float(np.mean([r["safety_burden"]>.5 for r in rows]))
    safety_gate = metrics["safety_burden"]["mean"] <= .1 and safety_rate <= .05
    all_gate = bool(coverage and incidence and trigger_gate and oracle_gate and selector_gate and fraction_gate and safety_gate)
    result = {"scope":"Frozen cross-corpus KRAISLER P3-R transfer; corpus previously used for another endpoint",
              "coverage":{"eligible_tracks":len(cases),"used_tracks":len(set(r['track'] for r in rows)),"interventions":len(rows)},
              "incidence":{"events":int(y.sum()),"affected_tracks":affected_tracks,"prevalence":prevalence},
              "trigger":{"auroc":auc,"average_precision":ap,"recall":recall,"false_positive_rate":fpr,
                         "predicted_positive":int(pred.sum()),"frozen_threshold":model["threshold"]},
              "metrics":metrics,"safety_violation_rate":safety_rate,
              "gates":{"coverage":coverage,"incidence":incidence,"trigger":bool(trigger_gate),"oracle":bool(oracle_gate),
                       "selector":bool(selector_gate),"fraction_of_oracle":bool(fraction_gate),"safety":bool(safety_gate),"all":all_gate},
              "decision":"eligible_for_preserved_holdout" if all_gate else "stop_before_preserved_holdout",
              "rows":rows}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__ == "__main__": main()
