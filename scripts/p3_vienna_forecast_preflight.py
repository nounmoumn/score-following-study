#!/usr/bin/env python3
"""Outcome-blind Vienna 4x22 clock audit and fixed forecast split."""

from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/vienna4x22_rematched";OUT=ROOT/"results/p3_vienna_forecast_preflight"
LINE=re.compile(r'^snote\([^,]+,.*?,(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),\[[^]]*\]\)-note\(.*?,(\d+),(\d+),(\d+),(\d+)\)\.$')
SPLIT={"Chopin_op10_no3":"calibration","Mozart_K331_1st-mov":"calibration",
       "Chopin_op38":"validation","Schubert_D783_no15":"validation"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_clock(path):
    text=path.read_text(encoding="utf8")
    units=int(re.search(r'info\(midiClockUnits,(\d+)\)',text).group(1))
    rate=int(re.search(r'info\(midiClockRate,(\d+)\)',text).group(1))
    scale=rate/(units*1_000_000); grouped=defaultdict(list)
    for line in text.splitlines():
        match=LINE.match(line)
        if match:grouped[float(match.group(1))].append(int(match.group(3))*scale)
    score=np.asarray(sorted(grouped),dtype=float)*.5
    performance=np.asarray([np.median(grouped[value/.5])*1. for value in score],dtype=float)
    if len(score)<2:raise ValueError(f"too few matched onsets: {path}")
    return score,performance

def main():
    rows=[]
    for match_path in sorted((BASE/"match").glob("*.match")):
        stem=match_path.stem;work=re.sub(r'_p\d+$','',stem)
        midi=BASE/"midi"/(stem+".mid");score=BASE/"musicxml"/(work+".musicxml")
        score_clock,perf_clock=load_clock(match_path)
        rows.append({"work":work,"stem":stem,"split":SPLIT[work],"matched_onsets":len(score_clock),
                     "score_clock_strict":bool(np.all(np.diff(score_clock)>0)),
                     "performance_clock_strict":bool(np.all(np.diff(perf_clock)>0)),
                     "clock_start_sec":float(perf_clock[0]),"clock_stop_sec":float(perf_clock[-1]),
                     "midi":str(midi.relative_to(ROOT)).replace('\\','/'),
                     "score":str(score.relative_to(ROOT)).replace('\\','/'),
                     "match":str(match_path.relative_to(ROOT)).replace('\\','/'),
                     "midi_sha256":sha(midi),"score_sha256":sha(score),"match_sha256":sha(match_path)})
    result={"scope":"forecast-position preflight only; no tracker or drift outcomes",
            "source_commit":"7b60448012c8b6bbf3e5045831bddc20d478db1a",
            "split_rule":SPLIT,"counts":{"performances":len(rows),"works":len({r['work'] for r in rows}),
            "calibration":sum(r['split']=='calibration' for r in rows),"validation":sum(r['split']=='validation' for r in rows)},
            "all_clocks_strict":all(r['score_clock_strict'] and r['performance_clock_strict'] for r in rows),"rows":rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=="__main__":main()
