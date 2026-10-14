#!/usr/bin/env python3
"""Verify the exact Lipschitz forecast-error bound after P3-FP failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_fp_forward_projection as fp
from score_following.temporal import healthy_before

CONFIG = ROOT / "configs/p3_temporal.yaml"; OUT = ROOT / "results/p3_fb_forecast_bound"


def help_bound(z, states, predicted_center, true_center, score_hop, cap=5.0):
    predicted_risk = np.minimum(np.abs(states-predicted_center)*score_hop, cap)
    true_risk = np.minimum(np.abs(states-true_center)*score_hop, cap)
    predicted_help = -float(predicted_risk @ z); true_help = -float(true_risk @ z)
    epsilon = abs(predicted_center-true_center)*score_hop
    bound = float(epsilon*np.abs(z).sum())
    return predicted_help, true_help, bound, float(abs(predicted_help-true_help)-bound)


def main():
    from p3_batik_external_run import load_gt
    from p3_kraisler_signal_run import load_case
    manifest=json.loads((ROOT/"results/p3_kraisler_signal/manifest.json").read_text(encoding="utf8"))
    cases=[row for row in manifest["rows"] if row["eligible"]]
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf8"));hop=float(cfg["hop_sec"])
    weights=tuple(float(x) for x in cfg["bayes"]["transition"])
    expected_step=float(sum(i*w for i,w in enumerate(weights))); rows=[]
    for case in cases:
        u,v,score_hop,duration=load_case(case,hop);beats,times=load_gt(ROOT/case["match"])
        truth=np.interp(np.arange(len(u))*hop,times,beats*.5)/score_hop
        _,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);path=trace.argmax(axis=1)
        error=np.abs(path-truth)*score_hop;states=np.arange(len(v))
        for time_sec in np.arange(15.,duration-5.+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+fp.FUTURE
            if endpoint>=len(error) or not healthy_before(error,end-1,fp.HISTORY,.5):continue
            z=fp.projected_action(trace,weights,end-fp.HISTORY,end,fp.FUTURE+1)
            center=float(path[end-1]+(fp.FUTURE+1)*expected_step)
            predicted,true,bound,violation=help_bound(z,states,center,truth[endpoint],score_hop)
            certified=abs(predicted)>bound
            rows.append({"track":case["track"],"time_sec":float(time_sec),"predicted_help":predicted,
                         "oracle_help":true,"realized_error_bound":bound,"violation":violation,
                         "retrospectively_certified":bool(certified),
                         "sign_match":bool(np.sign(predicted)==np.sign(true))})
    max_violation=max(row["violation"] for row in rows); certified=[row for row in rows if row["retrospectively_certified"]]
    result={"scope":"post-P3-FP mathematical audit; realized error is not online",
            "counts":{"landmarks":len(rows),"retrospectively_certified":len(certified)},
            "max_bound_violation":max_violation,
            "certification_share":len(certified)/len(rows),
            "certified_sign_agreement":float(np.mean([row["sign_match"] for row in certified])) if certified else None,
            "bound_verified":bool(max_violation<=1e-12 and all(row["sign_match"] for row in certified)),
            "online_p3_confirmed":False,"rows":rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":main()
