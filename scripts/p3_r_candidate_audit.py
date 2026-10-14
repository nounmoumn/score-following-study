"""Audit whether an intact recovery location survives in top posterior modes."""

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
from score_following.observation_evidence import predict_prior

OUT = ROOT / "results/p3_r_candidate"
CONFIG = ROOT / "configs/p3_asap_intervention.yaml"
TEMPORAL = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_R_CANDIDATE_PROTOCOL.md"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def posterior_until(cost, end, transition, beta, restart):
    posterior = np.zeros(cost.shape[1]); posterior[0] = 1
    path = []
    for i in range(end):
        if i: posterior = predict_prior(posterior, transition, restart)
        likelihood = np.exp(-beta * (cost[i] - cost[i].min()))
        posterior *= likelihood; posterior /= posterior.sum(); path.append(int(np.argmax(posterior)))
    return posterior, np.asarray(path)


def candidates(posterior, radius, separation, k):
    n = len(posterior); index = np.arange(n)
    prefix = np.concatenate(([0.], np.cumsum(posterior)))
    lo = np.maximum(0, index - radius); hi = np.minimum(n, index + radius + 1)
    masses = prefix[hi] - prefix[lo]
    available = np.ones(n, dtype=bool); selected = []
    for _ in range(k):
        valid = np.flatnonzero(available)
        if not len(valid): break
        center = int(valid[np.argmax(masses[valid])]); selected.append((center, float(masses[center])))
        available[np.abs(index - center) <= separation] = False
    return selected


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); temporal = yaml.safe_load(TEMPORAL.read_text(encoding="utf-8"))
    hop = float(cfg["hop_sec"]); onset = int(round(cfg["onset_sec"] / hop)); restore = int(round(cfg["restore_sec"] / hop))
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
            intact_path = bayes_from_cost(base, transition, 8., 1e-6)
            posterior, erased_prefix = posterior_until(erased, restore, transition, 8., 1e-6)
            erased_full = bayes_from_cost(erased, transition, 8., 1e-6)
            if not np.array_equal(erased_prefix, erased_full[:restore]): raise AssertionError("prefix state mismatch")
            radius = max(1, int(round(.5 / score_hop))); separation = max(radius + 1, int(round(2. / score_hop)))
            selected = candidates(posterior, radius, separation, 10)
            correct = int(intact_path[restore - 1])
            row = {"composition": item["composition"], "performance": item["performance"],
                   "score_hop_sec": score_hop, "intact_center": correct,
                   "erased_map": int(np.argmax(posterior)), "candidate_count": len(selected),
                   "correct_region_mass": float(posterior[max(0, correct-radius):min(len(posterior), correct+radius+1)].sum()),
                   "candidates": [{"center": c, "mass": m, "distance_score_sec": abs(c-correct)*score_hop}
                                  for c, m in selected]}
            for k in (1, 3, 5, 10):
                row[f"recall_at_{k}"] = any(abs(c - correct) <= radius for c, _ in selected[:k])
            rows.append(row); print(f"[{number}/{len(manifest)}] {item['performance']}")
    grouped = defaultdict(list)
    for row in rows: grouped[row["composition"]].append(row)
    rates = {f"performance_recall_at_{k}": sum(r[f"recall_at_{k}"] for r in rows) / len(rows) for k in (1,3,5,10)}
    rates.update({f"work_all_recall_at_{k}": sum(all(r[f"recall_at_{k}"] for r in rr) for rr in grouped.values()) / len(grouped)
                  for k in (1,3,5,10)})
    enough = all(r["candidate_count"] >= 5 for r in rows)
    gate = rates["performance_recall_at_5"] >= .8 and rates["work_all_recall_at_5"] >= .8 and enough
    result = {"scope": "ASAP development candidate coverage; not recovery validation",
              "performances": len(rows), "works": len(grouped), **rates,
              "all_have_five_candidates": enough, "gate_pass": gate,
              "decision": "advance_to_oracle_and_selector" if gate else "candidate_loss_bottleneck",
              "rows": rows,
              "provenance": {"protocol_sha256": sha256(PROTOCOL), "config_sha256": sha256(CONFIG),
                             "temporal_sha256": sha256(TEMPORAL), "script_sha256": sha256(Path(__file__))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
