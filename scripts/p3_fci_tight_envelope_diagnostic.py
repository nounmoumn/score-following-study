#!/usr/bin/env python3
"""Post-P3-FC diagnostic of the exact positional help envelope."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fc_fresh_certificate as fc
import p3_fp_forward_projection as fp
from score_following.temporal import healthy_before
OUT=ROOT/"results/p3_fci_tight_envelope_diagnostic"

def help_envelope(z,center,radius_sec,score_hop,cap=5.):
    states=np.arange(len(z),dtype=float);lo=max(0.,center-radius_sec/score_hop);hi=min(len(z)-1.,center+radius_sec/score_hop)
    cap_states=cap/score_hop;knots=[lo,hi]
    for values in (states,states-cap_states,states+cap_states):
        knots.extend(values[(values>=lo)&(values<=hi)].tolist())
    knots=np.unique(knots);helps=np.asarray([-np.minimum(np.abs(states-g)*score_hop,cap)@z for g in knots])
    return float(helps.min()),float(helps.max())

def main():
    manifest=json.loads(fc.MANIFEST.read_text());cases=[r for r in manifest["rows"] if r["eligible"]]
    cfg=yaml.safe_load((ROOT/"configs/p3_temporal.yaml").read_text());hop=float(cfg["hop_sec"]);weights=tuple(float(x) for x in cfg["bayes"]["transition"])
    rows=[]
    for number,case in enumerate(cases,1):
        u,v,score_hop,truth,duration=fc.load_case(case,hop);_,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);path=trace.argmax(axis=1);error=np.abs(path-truth)*score_hop;states=np.arange(len(v))
        for time_sec in np.arange(15.,duration-2.1+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+20
            if endpoint>=len(trace) or not healthy_before(error,end-1,20,.5):continue
            predicted=trace[end-1].copy()
            for _ in range(21):predicted=dc.apply_prediction(predicted,weights)
            center=fc.median_state(predicted);covered=abs(center-truth[endpoint])*score_hop<=fc.RADIUS
            z=fp.projected_action(trace,weights,end-20,end,21);low,high=help_envelope(z,center,fc.RADIUS,score_hop)
            certified=low>0 or high<0;oracle=-float(np.minimum(np.abs(states-truth[endpoint])*score_hop,5.)@z)
            sign=1 if low>0 else -1 if high<0 else 0
            rows.append({"source":case["source"],"unit":case["unit"],"covered":covered,"certified":certified,
                         "certificate_sign":sign,"oracle_sign":int(np.sign(oracle)),"sign_match":sign==np.sign(oracle)})
        print(f"[{number}/11] {case['source']} {case['unit']}")
    certified=[r for r in rows if r["certified"]];covered_cert=[r for r in certified if r["covered"]]
    result={"scope":"post-failure tight-envelope diagnostic; cannot qualify on opened cohort","landmarks":len(rows),
            "certificate_share":len(certified)/len(rows),"certified":len(certified),"covered_certified":len(covered_cert),
            "covered_certificate_sign_accuracy":float(np.mean([r["sign_match"] for r in covered_cert])) if covered_cert else None,
            "source_certificate_share":{s:float(np.mean([r["certified"] for r in rows if r["source"]==s])) for s in ("KRAISLER","Batik")},
            "requires_new_validation":True,"natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
