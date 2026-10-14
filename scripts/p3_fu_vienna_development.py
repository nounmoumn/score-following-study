#!/usr/bin/env python3
"""Develop one robust-velocity future-position radius on opened Vienna data."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_d_dose_response as dose
import p3_vienna_forecast_validation as vienna
from score_following.temporal import healthy_before
OUT=ROOT/"results/p3_fu_vienna_development"

def higher(values,q):return float(np.percentile(np.asarray(values,dtype=float),q,method="higher"))
def forecast_center(trace,end,history=20,future=21):
    states=np.arange(trace.shape[1]);means=trace@states
    velocity=float(np.clip(np.median(np.diff(means[end-history:end])),0.,3.))
    return float(means[end-1]+future*velocity),velocity

def main():
    manifest=json.loads(vienna.MANIFEST.read_text());cfg=yaml.safe_load(vienna.CONFIG.read_text());hop=float(cfg["hop_sec"])
    weights=tuple(float(x) for x in cfg["bayes"]["transition"]);cache={};rows=[]
    for number,case in enumerate(manifest["rows"],1):
        u,v,score_hop,truth,available=vienna.load_case(case,hop,cache)
        _,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);path=trace.argmax(axis=1);error=np.abs(path-truth)*score_hop
        for time_sec in np.arange(15.,available-2.1+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+20
            if endpoint>=len(trace) or not healthy_before(error,end-1,20,.5):continue
            center,velocity=forecast_center(trace,end)
            rows.append({"stem":case["stem"],"work":case["work"],"error_sec":abs(center-truth[endpoint])*score_hop,"velocity":velocity})
        print(f"[{number}/88] {case['stem']}")
    by={stem:[r for r in rows if r["stem"]==stem] for stem in sorted({r["stem"] for r in rows})}
    per={stem:higher([r["error_sec"] for r in values],95) for stem,values in by.items() if values}
    radius=higher(list(per.values()),95)
    gates={"healthy_performances_ge_70":len(per)>=70,"landmarks_ge_500":len(rows)>=500,"radius_le_0_75":radius<=.75}
    result={"scope":"opened Vienna position-only development","counts":{"healthy_performances":len(per),"landmarks":len(rows)},
            "radius_sec":radius,"median_velocity":float(np.median([r["velocity"] for r in rows])),
            "gates":{k:bool(v) for k,v in gates.items()},"fresh_transfer_authorized":bool(all(gates.values())),"natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
