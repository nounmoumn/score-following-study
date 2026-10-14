#!/usr/bin/env python3
"""Frozen KRAISLER development test of online forward-projected observation help."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_k_correction_deficit as inference
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_fp_forward_projection"
HISTORY = 20; FUTURE = 20; CAP = 5.0; SEED = 20260921


def projected_action(trace, weights, start, end, future_frames):
    z = np.zeros(trace.shape[1])
    for frame in range(start, end):
        prior = dc.apply_prediction(trace[frame - 1], weights)
        z = dc.apply_prediction(z, weights) + (trace[frame] - prior)
    for _ in range(future_frames):
        z = dc.apply_prediction(z, weights)
    return z


def main():
    from p3_batik_external_run import load_gt
    from p3_kraisler_signal_run import load_case
    manifest = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")); hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    expected_step = float(sum(step * weight for step, weight in enumerate(weights)))
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, duration = load_case(case, hop)
        beats, perf_times = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(u)) * hop, perf_times, beats * .5) / score_hop
        cost = dose.cosine_cost(u, v); path, trace = dose.bayes_trace(cost, weights)
        error = np.abs(path - truth) * score_hop; states = np.arange(len(v))
        for time_sec in np.arange(15., duration - 5. + 1e-9, 5.):
            end = int(round(time_sec / hop)); endpoint = end + FUTURE
            if endpoint >= len(error) or not healthy_before(error, end - 1, HISTORY, .5): continue
            z = projected_action(trace, weights, end - HISTORY, end, FUTURE + 1)
            predicted_center = float(path[end - 1] + (FUTURE + 1) * expected_step)
            proxy_risk = np.minimum(np.abs(states - predicted_center) * score_hop, CAP)
            oracle_risk = np.minimum(np.abs(states - truth[endpoint]) * score_hop, CAP)
            proxy_help = -float(proxy_risk @ z); oracle_help = -float(oracle_risk @ z)
            onset = sustained_start(error, end, endpoint)
            rows.append({"track": case["track"], "time_sec": float(time_sec),
                         "proxy_help": proxy_help, "oracle_help": oracle_help,
                         "deficit": -proxy_help, "event": bool(onset is not None),
                         "onset_sec": None if onset is None else float(onset * hop)})
        print(f"[{number}/{len(cases)}] {case['track']}")
    proxy = np.asarray([row["proxy_help"] for row in rows]); oracle = np.asarray([row["oracle_help"] for row in rows])
    rho = float(spearmanr(proxy, oracle).statistic)
    sign_agreement = float(np.mean(np.sign(proxy) == np.sign(oracle)))
    event = [row["deficit"] for row in rows if row["event"]]
    nonevent = [row["deficit"] for row in rows if not row["event"]]
    auc = inference.rank_auc_safe(event, nonevent); rng = np.random.default_rng(SEED)
    auc_ci = inference.cluster_bootstrap_auc(rows, rng)
    null = np.asarray([inference.shifted_auc(rows, rng) for _ in range(inference.PERMUTATIONS)])
    null = null[np.isfinite(null)]; shift_p = float((1 + np.sum(null >= auc)) / (len(null) + 1))
    event_onsets = {(row["track"], row["onset_sec"]) for row in rows if row["event"]}
    event_tracks = {row["track"] for row in rows if row["event"]}
    metrics = {"proxy_oracle_spearman": rho, "proxy_oracle_sign_agreement": sign_agreement,
               "event_deficit_median": float(np.median(event)),
               "nonevent_deficit_median": float(np.median(nonevent)), "rank_auc": auc,
               "track_bootstrap_auc_95": auc_ci, "circular_shift_p": shift_p}
    gates = {"coverage": len(cases)==13 and len(rows)>=100 and len(event_onsets)>=5 and len(event_tracks)>=5,
             "fidelity_rho_ge_0_70": rho>=.70, "fidelity_sign_ge_0_75": sign_agreement>=.75,
             "event_median_gt_nonevent": np.median(event)>np.median(nonevent), "auc_ge_0_70": auc>=.70,
             "bootstrap_lower_gt_0_55": auc_ci[0]>.55, "circular_shift_p_le_0_05": shift_p<=.05}
    result = {"hypothesis":"P3-FP online forward-projected observation deficit",
              "counts":{"tracks":len(cases),"landmarks":len(rows),"event_onsets":len(event_onsets),"event_tracks":len(event_tracks)},
              "metrics":metrics,"gates":{k:bool(v) for k,v in gates.items()},
              "development_pass":bool(all(gates.values())),"external_transfer_authorized":bool(all(gates.values())),
              "original_natural_p3_confirmed":False,"rows":rows}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__": main()
