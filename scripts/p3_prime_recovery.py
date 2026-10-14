"""Factorial observation/restart recovery intervention for P3-prime."""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from score_following.prospective import bootstrap_mean

OUT = ROOT / "results/p3_prime_recovery"
CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_PRIME_RECOVERY_PROTOCOL.md"

CONDITIONS = {
    "natural_restore": (8., 1e-6),
    "observation_rescue": (16., 1e-6),
    "restart_rescue": (8., 1e-3),
    "combined_rescue": (16., 1e-3),
}


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def bayes_scheduled(cost, transition, base_beta, base_restart, recovery_beta, recovery_restart,
                    restore, rescue_end):
    m, n = cost.shape; posterior = np.zeros(n); posterior[0] = 1
    estimate = np.zeros(m, dtype=int)
    for i, row in enumerate(cost):
        restart = recovery_restart if restore <= i < rescue_end else base_restart
        beta = recovery_beta if restore <= i < rescue_end else base_beta
        if i:
            prior = np.zeros(n)
            for step, weight in enumerate(transition):
                if step == 0: prior += weight * posterior
                else:
                    prior[step:] += weight * posterior[:-step]
                    prior[-1] += weight * posterior[-step:].sum()
            posterior = (1 - restart) * prior + restart / n
        likelihood = np.exp(-beta * (row - row.min()))
        posterior *= likelihood; posterior /= posterior.sum()
        estimate[i] = int(np.argmax(posterior))
    return estimate


def first_recovery(distance, start, end, threshold=.5, run=5, hop=.1):
    healthy = distance[start:end] <= threshold
    for offset in range(max(0, len(healthy) - run + 1)):
        if np.all(healthy[offset:offset + run]): return offset * hop
    return (end - start) * hop


def summarize(values, seed):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "ci95": bootstrap_mean(values, seed, 10000),
            "positive": int(np.sum(values > 0)), "n": len(values)}


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); temporal = yaml.safe_load(TEMPORAL.read_text(encoding="utf-8"))
    hop = float(cfg["hop_sec"]); onset = int(round(cfg["onset_sec"] / hop)); restore = int(round(cfg["restore_sec"] / hop))
    recovery_end = int(round(cfg["recovery_end_sec"] / hop)); rescue_end = restore + int(round(5 / hop))
    transition = np.asarray(temporal["bayes"]["transition"], dtype=float)
    with (ROOT / cfg["manifest"]).open(encoding="utf-8", newline="") as handle: manifest = list(csv.DictReader(handle))
    root = ROOT / cfg["asap_root"]
    with (root / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = {r["midi_performance"]: r for r in csv.DictReader(handle)}
    rows = []
    with threadpool_limits(limits=1):
        for number, item in enumerate(manifest, 1):
            meta = metadata[item["performance"]]
            pnotes = load_notes(root / meta["midi_performance"]); snotes = load_notes(root / meta["midi_score"])
            pd, sd = max(n[1] for n in pnotes), max(n[1] for n in snotes); score_hop = hop * sd / pd
            u = notes_to_chroma_stream(pnotes, cfg["duration_sec"], hop)
            v = notes_to_chroma_stream(snotes, sd, score_hop); base = cosine_cost(u, v)
            erased = base.copy(); erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
            intact = bayes_from_cost(base, transition, 8., 1e-6)
            natural = bayes_from_cost(erased, transition, 8., 1e-6)
            paths = {"natural_restore": natural}
            for name, (beta, restart) in CONDITIONS.items():
                if name == "natural_restore": continue
                paths[name] = bayes_scheduled(erased, transition, 8., 1e-6, beta, restart, restore, rescue_end)
            if any(not np.array_equal(path[:restore], natural[:restore]) for name, path in paths.items() if name != "natural_restore"):
                raise AssertionError("rescue changed pre-restore state")
            if not np.array_equal(natural, bayes_scheduled(erased, transition, 8., 1e-6, 8., 1e-6, restore, rescue_end)):
                raise AssertionError("scheduled natural path mismatch")
            distance = {name: np.abs(path - intact) * score_hop for name, path in paths.items()}
            radius = max(len(transition) - 1, int(round(.5 / score_hop)))
            intact_switches = int(np.sum(np.abs(np.diff(intact[onset - 1:restore])) > radius))
            erased_switches = int(np.sum(np.abs(np.diff(natural[onset - 1:restore])) > radius))
            row = {"composition": item["composition"], "performance": item["performance"],
                   "pre_max_distance": float(np.max(distance["natural_restore"][:onset])),
                   "erasure_distance": float(distance["natural_restore"][onset:restore].mean()),
                   "nonlocal_switch_increase": erased_switches - intact_switches,
                   "natural_recovery_burden": float(distance["natural_restore"][restore:recovery_end].mean()),
                   "natural_recovery_time": first_recovery(distance["natural_restore"], restore, recovery_end, hop=hop)}
            for name in ("observation_rescue", "restart_rescue", "combined_rescue"):
                row[name + "_burden"] = float(distance[name][restore:recovery_end].mean())
                row[name + "_improvement"] = row["natural_recovery_burden"] - row[name + "_burden"]
                row[name + "_time"] = first_recovery(distance[name], restore, recovery_end, hop=hop)
            rows.append(row); print(f"[{number}/{len(manifest)}] {item['performance']}")
    grouped = defaultdict(list)
    for row in rows: grouped[row["composition"]].append(row)
    works = []
    fields = ["erasure_distance", "nonlocal_switch_increase", "natural_recovery_burden",
              "observation_rescue_improvement", "restart_rescue_improvement", "combined_rescue_improvement"]
    for composition, rr in sorted(grouped.items()):
        works.append({"composition": composition, **{f: float(np.mean([r[f] for r in rr])) for f in fields}})
    seed = 20260920
    metrics = {field: summarize([r[field] for r in works], seed) for field in fields}
    natural_mean = metrics["natural_recovery_burden"]["mean"]
    combined = metrics["combined_rescue_improvement"]
    relative = combined["mean"] / natural_mean if natural_mean else 0.
    removal_ok = metrics["erasure_distance"]["mean"] > 0 and metrics["erasure_distance"]["ci95"][0] > 0
    switch_ok = metrics["nonlocal_switch_increase"]["mean"] > 0 and metrics["nonlocal_switch_increase"]["ci95"][0] > 0
    combined_ok = combined["mean"] > 0 and combined["ci95"][0] > 0 and relative >= .2
    singles = {name: metrics[name + "_rescue_improvement"]["mean"] > 0 and metrics[name + "_rescue_improvement"]["ci95"][0] > 0
               for name in ("observation", "restart")}
    result = {"scope": "ASAP development recovery intervention; not natural P3 or independent confirmation",
              "performances": len(rows), "works": len(works), "metrics": metrics,
              "combined_relative_burden_reduction": relative,
              "component_gates": {"removal": removal_ok, "nonlocal_switch": switch_ok,
                                  "combined_rescue": combined_ok, **singles},
              "gate_pass": bool(removal_ok and switch_ok and combined_ok and any(singles.values())),
              "decision": "transfer_frozen_design" if removal_ok and switch_ok and combined_ok and any(singles.values()) else "partial_or_failed",
              "rows": rows, "works_rows": works,
              "provenance": {"protocol_sha256": sha256(PROTOCOL), "config_sha256": sha256(CONFIG),
                             "temporal_sha256": sha256(TEMPORAL), "script_sha256": sha256(Path(__file__))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "works_rows", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
