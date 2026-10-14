#!/usr/bin/env python3
"""Replay the exact P3-DC identity on admitted branches in three corpora."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_tdr_cp_wtc_validation as cp
import p3_pt_batik_heterogeneity as batik
from p3_batik_external_run import load_gt
from p3_kraisler_signal_run import load_case as load_kraisler
from score_following.temporal import healthy_before

CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_dc_corpus_audit"
DURATIONS = (2, 5, 10)


def audit_cases(name, cases, loader, landmark_fn, hop, weights):
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth, available = loader(case, hop)
        cost = dose.cosine_cost(u, v); _, trace = dose.bayes_trace(cost, weights)
        error = np.abs(trace.argmax(axis=1) - truth) * score_hop
        for time_sec in landmark_fn(case, available):
            end = int(round(time_sec / hop))
            if end + 104 >= len(cost) or not healthy_before(error, end - 1, 20, .5):
                continue
            for seconds in DURATIONS:
                stop = end + int(round(seconds / hop)) + 4
                result = dc.decompose(trace[end - 1], cost[end:stop], truth[end:stop], score_hop, weights)
                rows.append({"corpus": name, "unit": case["unit"], "time_sec": float(time_sec),
                             "duration_sec": seconds, **{key: result[key] for key in
                             ("posterior_residual", "risk_residual", "erasure_effect", "projected_help_sum")}})
        print(f"{name} [{number}/{len(cases)}] {case['unit']}")
    return rows


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")); hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows = []
    km = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text(encoding="utf8"))
    kcases = [dict(row, unit=row["track"]) for row in km["rows"] if row["eligible"]]
    def kl(case, local_hop):
        u, v, score_hop, available = load_kraisler(case, local_hop)
        beats, times = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(u)) * local_hop, times, beats * .5) / score_hop
        return u, v, score_hop, truth, available
    rows += audit_cases("KRAISLER", kcases, kl,
                        lambda case, available: np.arange(15., available - 10.4 + 1e-9, 10.), hop, weights)

    bm = json.loads((ROOT / "results/p3_batik_external/manifest.json").read_text(encoding="utf8"))
    bcases = [dict(row, unit=f"{row['sonata']}-{row['movement']}") for row in bm["rows"] if row["eligible"]]
    def bl(case, local_hop):
        u, v, score_hop, truth = batik.load_case(case, local_hop)
        return u, v, score_hop, truth, min(90., float(case["duration_sec"]))
    rows += audit_cases("Batik", bcases, bl,
                        lambda case, available: np.arange(15., available - 10.4 + 1e-9, 10.), hop, weights)

    cm = json.loads((ROOT / "results/p3_cp_wtc_preflight/manifest.json").read_text(encoding="utf8"))
    ccases = [dict(row, unit=row["piece"]) for row in cm["rows"] if row["eligible"]]
    rows += audit_cases("CP-WTC", ccases, cp.load_case,
                        lambda case, available: np.asarray([15.]), hop, weights)

    max_p = max(row["posterior_residual"] for row in rows)
    max_r = max(row["risk_residual"] for row in rows)
    mismatches = sum(np.sign(row["erasure_effect"]) != np.sign(row["projected_help_sum"]) for row in rows)
    by_corpus = {name: {"branches": sum(row["corpus"] == name for row in rows),
                        "positive_erasure_effect_share": float(np.mean([row["erasure_effect"] > 0 for row in rows if row["corpus"] == name]))}
                 for name in ("KRAISLER", "Batik", "CP-WTC")}
    result = {"counts": {"branches": len(rows)}, "by_corpus": by_corpus,
              "max_posterior_residual": max_p, "max_risk_residual": max_r,
              "sign_mismatches": int(mismatches),
              "passed": bool(max_p <= 1e-12 and max_r <= 1e-12 and mismatches == 0),
              "natural_temporal_p3_confirmed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
