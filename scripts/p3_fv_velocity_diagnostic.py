#!/usr/bin/env python3
"""Single post-failure diagnostic using robust recent posterior velocity."""

from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_d_dose_response as dose
import p3_fp_forward_projection as fp
import p3_k_correction_deficit as inference
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before


def robust_velocity(trace,start,end):
    mean_state=trace@np.arange(trace.shape[1])
    return float(np.clip(np.median(np.diff(mean_state[start:end])),0.,3.))


def main():
    from p3_batik_external_run import load_gt
    from p3_kraisler_signal_run import load_case
    cfg=yaml.safe_load((ROOT/"configs/p3_temporal.yaml").read_text());hop=float(cfg["hop_sec"])
    weights=tuple(float(x) for x in cfg["bayes"]["transition"])
    manifest=json.loads((ROOT/"results/p3_kraisler_signal/manifest.json").read_text())
    rows=[]
    for case in (row for row in manifest["rows"] if row["eligible"]):
        u,v,score_hop,duration=load_case(case,hop);beats,times=load_gt(ROOT/case["match"])
        truth=np.interp(np.arange(len(u))*hop,times,beats*.5)/score_hop
        _,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);path=trace.argmax(axis=1)
        error=np.abs(path-truth)*score_hop;states=np.arange(len(v))
        for time_sec in np.arange(15.,duration-5.+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+fp.FUTURE
            if endpoint>=len(error) or not healthy_before(error,end-1,fp.HISTORY,.5):continue
            z=fp.projected_action(trace,weights,end-fp.HISTORY,end,fp.FUTURE+1)
            velocity=robust_velocity(trace,end-fp.HISTORY,end)
            current_mean=float(trace[end-1]@states);center=current_mean+(fp.FUTURE+1)*velocity
            proxy=-float(np.minimum(np.abs(states-center)*score_hop,5.)@z)
            oracle=-float(np.minimum(np.abs(states-truth[endpoint])*score_hop,5.)@z)
            onset=sustained_start(error,end,endpoint)
            rows.append({"track":case["track"],"proxy_help":proxy,"oracle_help":oracle,
                         "deficit":-proxy,"event":bool(onset is not None),"velocity":velocity})
    proxy=np.array([r["proxy_help"] for r in rows]);oracle=np.array([r["oracle_help"] for r in rows])
    event=[r["deficit"] for r in rows if r["event"]];nonevent=[r["deficit"] for r in rows if not r["event"]]
    result={"scope":"post-failure exploratory diagnostic; no success gate or transfer authorization",
            "landmarks":len(rows),"proxy_oracle_spearman":float(spearmanr(proxy,oracle).statistic),
            "sign_agreement":float(np.mean(np.sign(proxy)==np.sign(oracle))),
            "event_auc":inference.rank_auc_safe(event,nonevent),
            "event_median_deficit":float(np.median(event)),"nonevent_median_deficit":float(np.median(nonevent)),
            "median_velocity_states_per_frame":float(np.median([r["velocity"] for r in rows])),
            "p3_confirmed":False}
    out=ROOT/"results/p3_fv_velocity_diagnostic";out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
