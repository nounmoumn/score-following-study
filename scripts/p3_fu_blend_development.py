#!/usr/bin/env python3
"""Final Vienna-only shrinkage selection before any fresh-cohort outcome."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_d_dose_response as dose
import p3_fu_vienna_development as velocity
import p3_vienna_forecast_validation as vienna
from score_following.temporal import healthy_before
OUT=ROOT/"results/p3_fu_blend_development";ALPHAS=(0.,.25,.5,.75,1.)

def higher(values,q):return float(np.percentile(np.asarray(values,dtype=float),q,method="higher"))

def main():
    manifest=json.loads(vienna.MANIFEST.read_text());cfg=yaml.safe_load(vienna.CONFIG.read_text());hop=float(cfg["hop_sec"])
    weights=tuple(float(x) for x in cfg["bayes"]["transition"]);cache={};rows=[]
    for number,case in enumerate(manifest["rows"],1):
        u,v,score_hop,truth,available=vienna.load_case(case,hop,cache)
        _,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);path=trace.argmax(axis=1);error=np.abs(path-truth)*score_hop;states=np.arange(len(v));means=trace@states
        for time_sec in np.arange(15.,available-2.1+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+20
            if endpoint>=len(trace) or not healthy_before(error,end-1,20,.5):continue
            recent=float(np.clip(np.median(np.diff(means[end-20:end])),0.,3.))
            row={"stem":case["stem"],"work":case["work"]}
            for alpha in ALPHAS:
                center=float(means[end-1]+21*((1-alpha)+alpha*recent))
                row[str(alpha)]=abs(center-truth[endpoint])*score_hop
            rows.append(row)
        print(f"[{number}/88] {case['stem']}")
    summaries=[]
    for alpha in ALPHAS:
        key=str(alpha);work_radii={}
        for work in sorted({r["work"] for r in rows}):
            perf=[]
            for stem in sorted({r["stem"] for r in rows if r["work"]==work}):
                perf.append(higher([r[key] for r in rows if r["stem"]==stem],95))
            work_radii[work]=higher(perf,95)
        summaries.append({"alpha":alpha,"work_radii":work_radii,"worst_work_radius":max(work_radii.values())})
    chosen=min(summaries,key=lambda r:(r["worst_work_radius"],r["alpha"]))
    healthy=len({r["stem"] for r in rows});authorized=healthy==88 and len(rows)>=1000 and chosen["worst_work_radius"]<=.75
    result={"scope":"opened Vienna position-only final shrinkage development","counts":{"healthy_performances":healthy,"landmarks":len(rows)},
            "candidates":summaries,"selected":chosen,"fresh_transfer_authorized":authorized,"natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
