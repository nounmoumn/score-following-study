#!/usr/bin/env python3
"""Frozen KRAISLER then Batik transfer of P3-TD duration response."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_i_natural_equivalence as natural
import p3_p_healthy_erasure as erasure
import p3_td_duration_response as duration
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

CONFIG = ROOT / "configs/p3_temporal.yaml"


def run_cases(cases, loader, unit_key, landmarks_for_case, out, scope, coverage_gate):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")); hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth, available_duration = loader(case, hop)
        cost = dose.cosine_cost(u, v); _, trace = dose.bayes_trace(cost, weights)
        full_error = np.abs(trace.argmax(axis=1)-truth)*score_hop
        unit = case[unit_key]
        for time_sec in landmarks_for_case(case, available_duration):
            end=int(round(time_sec/hop)); complete_stop=end+100+duration.CONFIRM
            if complete_stop >= len(cost) or not healthy_before(full_error,end-1,20,.5): continue
            for seconds in duration.DURATIONS:
                frames=int(round(seconds/hop)); stop=end+frames+duration.CONFIRM
                intact=natural.replay(trace[end-1],cost[end:stop],weights)
                erased_cost=np.broadcast_to(cost[end:stop].mean(axis=1,keepdims=True),cost[end:stop].shape)
                erased=natural.replay(trace[end-1],erased_cost,weights)
                local_truth=truth[end:stop]
                intact_risk=erasure.expected_risk(intact,local_truth,score_hop)
                erased_risk=erasure.expected_risk(erased,local_truth,score_hop)
                intact_error=np.abs(intact.argmax(axis=1)-local_truth)*score_hop
                erased_error=np.abs(erased.argmax(axis=1)-local_truth)*score_hop
                rows.append({unit_key:unit,"time_sec":float(time_sec),"duration_sec":seconds,
                    "risk_effect":float(np.mean(erased_risk[frames-5:frames]-intact_risk[frames-5:frames])),
                    "intact_event":int(sustained_start(intact_error,0,frames) is not None),
                    "erased_event":int(sustained_start(erased_error,0,frames) is not None),
                    "intact_replay_residual":float(np.max(np.abs(intact-trace[end:stop]))),
                    "erased_row_range":float(np.max(np.ptp(erased_cost,axis=1)))})
        print(f"[{number}/{len(cases)}] {unit}")
    doses,effects,curve,risk_ci,incidence_ci,gates=duration.evaluate(rows,unit_key,coverage_gate)
    result={"hypothesis":"P3-TD healthy-state erasure duration response","scope":scope,
            "counts":{"units":len(effects),"landmarks":len(rows)//3,"runs":len(rows)},
            "mean_curve":curve,"ten_sec_risk_bootstrap_95":risk_ci,
            "ten_sec_incidence_bootstrap_95":incidence_ci,
            "positive_risk_rho_share":float(np.mean([row["risk_duration_rho"]>0 for row in effects])),
            "gates":{key:bool(value) for key,value in gates.items()},"transfer_pass":bool(all(gates.values())),
            "original_natural_p3_confirmed":False}
    out.mkdir(parents=True,exist_ok=True)
    for filename,data in (("runs.csv",rows),("unit_doses.csv",doses),("unit_effects.csv",effects)):
        with (out/filename).open("w",encoding="utf8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2)); return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--corpus",choices=("kraisler","batik"),required=True)
    args=parser.parse_args()
    from p3_batik_external_run import load_gt
    if args.corpus=="kraisler":
        from p3_kraisler_signal_run import load_case as load_kraisler
        manifest=json.loads((ROOT/"results/p3_kraisler_signal/manifest.json").read_text(encoding="utf8"))
        cases=[dict(row,unit=row["track"]) for row in manifest["rows"] if row["eligible"]]
        def loader(case,hop):
            u,v,score_hop,available=load_kraisler(case,hop)
            beats,times=load_gt(ROOT/case["match"])
            truth=np.interp(np.arange(len(u))*hop,times,beats*.5)/score_hop
            return u,v,score_hop,truth,available
        run_cases(cases,loader,"unit",lambda case,available:np.arange(15.,available-10.4+1e-9,10.),
                  ROOT/"results/p3_td_kraisler_transfer","KRAISLER frozen transfer",
                  lambda units,landmarks:units==13 and landmarks>=50)
    else:
        prior=json.loads((ROOT/"results/p3_td_kraisler_transfer/result.json").read_text(encoding="utf8"))
        if not prior["transfer_pass"]: raise SystemExit("KRAISLER transfer failed; Batik not authorized")
        import p3_pt_batik_heterogeneity as batik
        manifest=json.loads((ROOT/"results/p3_batik_external/manifest.json").read_text(encoding="utf8"))
        cases=[dict(row,unit=f"{row['sonata']}-{row['movement']}") for row in manifest["rows"] if row["eligible"]]
        def loader(case,hop):
            u,v,score_hop,truth=batik.load_case(case,hop)
            return u,v,score_hop,truth,min(90.,float(case["duration_sec"]))
        run_cases(cases,loader,"unit",lambda case,available:np.arange(15.,available-10.4+1e-9,10.),
                  ROOT/"results/p3_td_batik_transfer","Batik frozen transfer after KRAISLER",
                  lambda units,landmarks:units>=15 and landmarks>=90)


if __name__=="__main__":main()

