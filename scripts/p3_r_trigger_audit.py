#!/usr/bin/env python3
"""Leave-one-work-out audit of GT-free persistent-fault features."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost, bayes_from_cost
from p3_r_candidate_audit import candidates, posterior_until
from p3_r_basin_recovery import burden, likelihood

CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_r_trigger"
BETA, RESTART, K, BASIN_SECONDS = 8.0, 1e-6, 10, 2.0
WINDOWS_SEC = ((10, 20), (20, 30), (30, 40), (40, 50))
FEATURES = (
    "restore_entropy", "restore_max_probability", "restore_std_seconds",
    "decision_entropy", "decision_max_probability", "decision_std_seconds",
    "erased_net_progress_seconds", "erased_unique_fraction", "erased_nonlocal_switches",
    "candidate_max_mass", "candidate_mass_entropy", "max_log_bf", "score_margin",
    "selected_null_distance_seconds", "candidate_endpoint_spread_seconds",
)


def posterior_stats(posterior, score_hop):
    positive = posterior[posterior > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(posterior))
    index = np.arange(len(posterior), dtype=float)
    mean = float(np.sum(index * posterior))
    std = float(np.sqrt(np.sum((index - mean) ** 2 * posterior))) * score_hop
    return entropy, float(posterior.max()), std


def centers_and_masses(posterior, score_hop):
    radius = max(1, int(round(.5 / score_hop)))
    separation = max(radius + 1, int(round(2 / score_hop)))
    selected = candidates(posterior, radius, separation, K)
    return [x[0] for x in selected], np.asarray([x[1] for x in selected])


def compare_features(cost, posterior, centers, transition, score_hop, restore, decision):
    radius, n = max(1, int(round(BASIN_SECONDS / score_hop))), len(posterior)
    states = []
    for center in centers:
        lo, hi = max(0, center-radius), min(n, center+radius+1)
        state = np.zeros_like(posterior); state[lo:hi] = posterior[lo:hi]
        state /= state.sum(); states.append(state)
    scores = np.zeros(len(states)); null_state = posterior.copy(); null_score = 0.
    for frame in range(restore, decision):
        emit = likelihood(cost[frame])
        prior = predict_prior(null_state, transition, 0.)
        weighted = prior * emit; z = float(weighted.sum())
        null_state = weighted / z; null_score += math.log(z)
        for j, state in enumerate(states):
            prior = predict_prior(state, transition, 0.)
            weighted = prior * emit; z = float(weighted.sum())
            states[j] = weighted / z; scores[j] += math.log(z)
    endpoints = np.asarray([np.argmax(s) for s in states])
    order = np.argsort(scores)
    selected = int(order[-1])
    return states, null_state, scores, endpoints, selected, null_score


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    temporal = yaml.safe_load(TEMPORAL.read_text(encoding="utf-8"))
    hop = float(cfg["hop_sec"]); transition = np.asarray(temporal["bayes"]["transition"])
    with (ROOT / cfg["manifest"]).open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    asap_root = ROOT / cfg["asap_root"]
    with (asap_root / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {r["midi_performance"]: r for r in csv.DictReader(handle)}
    rows = []
    with threadpool_limits(limits=1):
        for number, item in enumerate(manifest, 1):
            meta = metadata[item["performance"]]
            pn, sn = load_notes(asap_root/meta["midi_performance"]), load_notes(asap_root/meta["midi_score"])
            pd, sd = max(n[1] for n in pn), max(n[1] for n in sn); score_hop = hop*sd/pd
            base = cosine_cost(notes_to_chroma_stream(pn, cfg["duration_sec"], hop),
                               notes_to_chroma_stream(sn, sd, score_hop))
            intact = bayes_from_cost(base, transition, BETA, RESTART)
            for start_sec, restore_sec in WINDOWS_SEC:
                onset, restore = int(start_sec/hop), int(restore_sec/hop)
                decision, eval_end = restore+int(2/hop), restore+int(10/hop)
                erased = base.copy(); erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
                natural = bayes_from_cost(erased, transition, BETA, RESTART)
                label = burden(natural, intact, score_hop, decision, eval_end) > .5
                posterior, path = posterior_until(erased, restore, transition, BETA, RESTART)
                centers, masses = centers_and_masses(posterior, score_hop)
                _, null, scores, endpoints, selected, null_score = compare_features(
                    erased, posterior, centers, transition, score_hop, restore, decision)
                re, rm, rs = posterior_stats(posterior, score_hop)
                de, dm, ds = posterior_stats(null, score_hop)
                ep = path[onset:restore]
                nonlocal_radius = max(len(transition)-1, int(round(.5/score_hop)))
                mass_prob = masses / masses.sum()
                mass_entropy = -float(np.sum(mass_prob * np.log(np.maximum(mass_prob, 1e-300)))) / math.log(len(mass_prob))
                ordered = np.sort(scores)
                features = {
                    "restore_entropy": re, "restore_max_probability": rm, "restore_std_seconds": rs,
                    "decision_entropy": de, "decision_max_probability": dm, "decision_std_seconds": ds,
                    "erased_net_progress_seconds": float((ep[-1]-ep[0])*score_hop),
                    "erased_unique_fraction": float(len(np.unique(ep))/len(ep)),
                    "erased_nonlocal_switches": int(np.sum(np.abs(np.diff(ep)) > nonlocal_radius)),
                    "candidate_max_mass": float(masses.max()), "candidate_mass_entropy": mass_entropy,
                    "max_log_bf": float(scores[selected]-null_score),
                    "score_margin": float(ordered[-1]-ordered[-2]),
                    "selected_null_distance_seconds": float(abs(endpoints[selected]-np.argmax(null))*score_hop),
                    "candidate_endpoint_spread_seconds": float((endpoints.max()-endpoints.min())*score_hop),
                }
                rows.append({"composition": item["composition"], "performance": item["performance"],
                             "window_sec": [start_sec, restore_sec], "persistent_event": bool(label), **features})
            print(f"[{number}/{len(manifest)}] {item['performance']}")

    x = np.asarray([[row[f] for f in FEATURES] for row in rows]); y = np.asarray([row["persistent_event"] for row in rows], dtype=int)
    groups = np.asarray([row["composition"] for row in rows]); scores = np.zeros(len(rows))
    for group in sorted(set(groups)):
        test = groups == group; train = ~test
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight="balanced", max_iter=5000, random_state=20260920))
        model.fit(x[train], y[train]); scores[test] = model.predict_proba(x[test])[:, 1]
    prevalence = float(y.mean()); auc = float(roc_auc_score(y, scores)); ap = float(average_precision_score(y, scores))
    threshold = float(np.quantile(scores[y == 0], .95, method="higher"))
    predicted = scores >= threshold
    recall = float(np.sum(predicted & (y == 1)) / np.sum(y == 1))
    false_positive_rate = float(np.sum(predicted & (y == 0)) / np.sum(y == 0))
    positive_events = int(np.sum(predicted & (y == 1)))
    gate = auc >= .75 and ap >= 2*prevalence and recall >= .40 and positive_events >= 10
    final_model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., class_weight="balanced", max_iter=5000, random_state=20260920))
    final_model.fit(x, y)
    final_scores = final_model.predict_proba(x)[:, 1]
    final_threshold = float(np.quantile(final_scores[y == 0], .95, method="higher"))
    scaler, logistic = final_model.named_steps["standardscaler"], final_model.named_steps["logisticregression"]
    frozen_model = {"feature_order": FEATURES, "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(), "coefficient": logistic.coef_[0].tolist(),
                    "intercept": float(logistic.intercept_[0]), "threshold": final_threshold}
    result = {"scope": "ASAP development LOCO trigger audit; not independent confirmation",
              "n": len(rows), "events": int(y.sum()), "works": len(set(groups)), "features": FEATURES,
              "model": "standardized balanced L2 logistic C=1",
              "metrics": {"prevalence": prevalence, "auroc": auc, "average_precision": ap,
                          "healthy_95pct_threshold": threshold, "recall_at_threshold": recall,
                          "false_positive_rate": false_positive_rate,
                          "positive_events_at_threshold": positive_events,
                          "total_predicted_positive": int(predicted.sum())},
              "frozen_full_development_model": frozen_model,
              "gate_pass": bool(gate),
              "decision": "freeze_for_external_transfer" if gate else "gt_free_trigger_not_supported",
              "rows": [{**row, "oof_score": float(score), "trigger": bool(p)} for row, score, p in zip(rows, scores, predicted)]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
