#!/usr/bin/env python3
"""Outcome-blind isotonic-clock salvage of never-opened direct-MIDI tracks."""
from __future__ import annotations
import hashlib,json,re,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATCHED=re.compile(r"^snote\(([^,]+),.*?,(-?\d+\.\d+),(-?\d+\.\d+),\[[^]]*\]\)-note\([^,]+,(\d+),(\d+),")
OUT=ROOT/"results/p3_clock_repair_preflight";MAX_CORRECTION=.15

def pava(values):
    values=np.asarray(values,dtype=float);means=[];weights=[];starts=[];ends=[]
    for i,value in enumerate(values):
        means.append(float(value));weights.append(1);starts.append(i);ends.append(i+1)
        while len(means)>=2 and means[-2]>means[-1]:
            w=weights[-2]+weights[-1];m=(means[-2]*weights[-2]+means[-1]*weights[-1])/w
            means[-2:]=[m];weights[-2:]=[w];ends[-2:]=[ends[-1]];starts[-2:]=[starts[-2]]
    result=np.empty(len(values))
    for mean,start,end in zip(means,starts,ends):result[start:end]=mean
    # Strict interpolation clock with negligible deterministic ordering.
    eps=1e-9;adjusted=result.copy()
    for i in range(1,len(adjusted)):
        if adjusted[i]<=adjusted[i-1]:adjusted[i]=adjusted[i-1]+eps
    return adjusted

def raw_clock(path):
    text=path.read_text(encoding="utf8");units=int(re.search(r'info\(midiClockUnits,(\d+)\)',text).group(1));rate=int(re.search(r'info\(midiClockRate,(\d+)\)',text).group(1))
    grouped=defaultdict(list)
    for line in text.splitlines():
        m=MATCHED.match(line)
        if m:grouped[float(m.group(2))].append(int(m.group(5))*rate/(units*1_000_000))
    beats=np.asarray(sorted(grouped));times=np.asarray([np.median(grouped[b]) for b in beats])
    return beats*.5,times

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def candidate(source,row):
    match=ROOT/row["match"];score_clock,raw=raw_clock(match);fixed=pava(raw);correction=np.abs(fixed-raw)
    return {"source":source,"unit":row.get("track") or f"{row['sonata']}-{row['movement']}",
            "performance":row["performance"],"score":row["score"],"match":row["match"],"duration_sec":row["duration_sec"],
            "anchors":len(raw),"raw_nonpositive_steps":int(np.sum(np.diff(raw)<=0)),
            "max_abs_clock_correction_sec":float(correction.max()),"mean_abs_clock_correction_sec":float(correction.mean()),
            "fixed_clock_strict":bool(np.all(np.diff(fixed)>0)),
            "eligible":bool(correction.max()<=MAX_CORRECTION and np.all(np.diff(fixed)>0)),
            "sha256":{"performance":sha(ROOT/row["performance"]),"score":sha(ROOT/row["score"]),"match":sha(match)}}

def main():
    km=json.loads((ROOT/"results/p3_kraisler_signal/manifest.json").read_text());bm=json.loads((ROOT/"results/p3_batik_external/manifest.json").read_text());rows=[]
    rows += [candidate("KRAISLER",r) for r in km["rows"] if not r["eligible"] and r["exclusion_reason"]=="non-monotone matched performance clock"]
    rows += [candidate("Batik",r) for r in bm["rows"] if not r["eligible"] and r["exclusion_reason"]=="non-monotone matched performance clock"]
    result={"scope":"outcome-blind clock repair only; selected tracker outcomes never opened","method":"PAVA L2 isotonic projection plus 1e-9 strict ordering",
            "max_correction_sec":MAX_CORRECTION,"counts":{"candidates":len(rows),"eligible":sum(r["eligible"] for r in rows),
            "kraisler":sum(r["eligible"] and r["source"]=="KRAISLER" for r in rows),"batik":sum(r["eligible"] and r["source"]=="Batik" for r in rows)},"rows":rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for row in rows:print(row["source"],row["unit"],row["eligible"],row["max_abs_clock_correction_sec"])

if __name__=="__main__":main()
