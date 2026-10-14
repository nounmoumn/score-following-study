#!/usr/bin/env python3
"""Calibrate then one-shot validate a two-second score-position interval."""

from __future__ import annotations
import csv,hashlib,json,sys,warnings
from pathlib import Path
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_vienna_forecast_preflight as preflight
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.temporal import healthy_before

MANIFEST=ROOT/"results/p3_vienna_forecast_preflight/manifest.json";CONFIG=ROOT/"configs/p3_temporal.yaml"
OUT=ROOT/"results/p3_vienna_forecast_validation";FUTURE=20;HISTORY=20

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def higher_percentile(values,q):return float(np.percentile(np.asarray(values,dtype=float),q,method="higher"))
def quantile_state(probability,q):return int(np.searchsorted(np.cumsum(probability),q,side="left"))

def score_note_cache(path,cache):
    if path not in cache:
        import partitura as pt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore");score=pt.load_musicxml(str(path))
            array=pt.score.unfold_part_maximal(score[0],ignore_leaps=False).note_array()
        cache[path]=[(float(r["onset_beat"])*.5,float(r["onset_beat"]+r["duration_beat"])*.5,int(r["pitch"])) for r in array]
    return cache[path]

def load_case(case,hop,cache):
    midi=ROOT/case["midi"];score_path=ROOT/case["score"];match=ROOT/case["match"]
    for path,key in ((midi,"midi_sha256"),(score_path,"score_sha256"),(match,"match_sha256")):
        if sha(path)!=case[key]:raise AssertionError(f"hash changed: {path}")
    score_clock,perf_clock=preflight.load_clock(match)
    score_hop=hop*(score_clock[-1]-score_clock[0])/(perf_clock[-1]-perf_clock[0])
    perf_notes=load_notes(midi);score_notes=score_note_cache(score_path,cache)
    perf_duration=max(n[1] for n in perf_notes);score_duration=max(n[1] for n in score_notes)
    u=notes_to_chroma_stream(perf_notes,perf_duration,hop);v=notes_to_chroma_stream(score_notes,score_duration,score_hop)
    truth=np.interp(np.arange(len(u))*hop,perf_clock,score_clock,left=score_clock[0],right=score_clock[-1])/score_hop
    return u,v,score_hop,truth,min(perf_duration,float(perf_clock[-1]))

def case_rows(case,hop,weights,cache):
    u,v,score_hop,truth,available=load_case(case,hop,cache)
    path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*score_hop
    rows=[]
    for time_sec in np.arange(15.,available-2.1+1e-9,5.):
        end=int(round(time_sec/hop));endpoint=end+FUTURE
        if endpoint>=len(trace) or not healthy_before(error,end-1,HISTORY,.5):continue
        predicted=trace[end-1].copy()
        for _ in range(FUTURE+1):predicted=dc.apply_prediction(predicted,weights)
        low=quantile_state(predicted,.005);center=quantile_state(predicted,.5);high=quantile_state(predicted,.995)
        true=float(truth[endpoint]);excess=max((low-true)*score_hop,(true-high)*score_hop,0.)
        rows.append({"work":case["work"],"stem":case["stem"],"split":case["split"],"time_sec":float(time_sec),
                     "score_hop":score_hop,"low_state":low,"center_state":center,"high_state":high,
                     "truth_state":true,"base_halfwidth_sec":max(center-low,high-center)*score_hop,
                     "excess_sec":float(excess),"path_consistent":bool(np.array_equal(path,trace.argmax(axis=1)))})
    return rows

def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf8"))
    if not manifest["all_clocks_strict"]:raise SystemExit("clock preflight failed")
    cfg=yaml.safe_load(CONFIG.read_text(encoding="utf8"));hop=float(cfg["hop_sec"])
    weights=tuple(float(x) for x in cfg["bayes"]["transition"]);rows=[];cache={}
    for number,case in enumerate(manifest["rows"],1):
        rows.extend(case_rows(case,hop,weights,cache));print(f"[{number}/88] {case['stem']}")
    calibration=[r for r in rows if r["split"]=="calibration"]
    by_perf={stem:[r for r in calibration if r["stem"]==stem] for stem in sorted({r["stem"] for r in calibration})}
    performance_q95={stem:higher_percentile([r["excess_sec"] for r in values],95) for stem,values in by_perf.items() if values}
    q=higher_percentile(list(performance_q95.values()),95)
    validation=[r for r in rows if r["split"]=="validation"]
    for row in validation:
        low=row["low_state"]*row["score_hop"]-q;high=row["high_state"]*row["score_hop"]+q
        truth=row["truth_state"]*row["score_hop"]
        row["calibrated_covered"]=bool(low<=truth<=high);row["calibrated_halfwidth_sec"]=row["base_halfwidth_sec"]+q
    healthy_perf={split:len({r["stem"] for r in rows if r["split"]==split}) for split in ("calibration","validation")}
    work_coverage={work:float(np.mean([r["calibrated_covered"] for r in validation if r["work"]==work])) for work in sorted({r["work"] for r in validation})}
    half=np.asarray([r["calibrated_halfwidth_sec"] for r in validation])
    gates={"manifest_44_44":manifest["counts"]["calibration"]==44 and manifest["counts"]["validation"]==44,
           "calibration_coverage":healthy_perf["calibration"]>=35 and len(calibration)>=200,
           "validation_coverage":healthy_perf["validation"]>=35 and len(validation)>=200,
           "q_le_0_75":q<=.75,"work_equal_coverage_ge_0_90":np.mean(list(work_coverage.values()))>=.90,
           "each_work_coverage_ge_0_85":min(work_coverage.values())>=.85,
           "median_halfwidth_le_1_0":np.median(half)<=1.,"p95_halfwidth_le_2_0":higher_percentile(half,95)<=2.,
           "all_paths_and_intervals_valid":all(r["path_consistent"] and r["low_state"]<=r["center_state"]<=r["high_state"] for r in rows)}
    result={"hypothesis":"two-second future score-position uncertainty qualification","counts":{"calibration_performances":healthy_perf["calibration"],
            "validation_performances":healthy_perf["validation"],"calibration_landmarks":len(calibration),"validation_landmarks":len(validation)},
            "calibration_radius_sec":q,"validation_work_coverage":work_coverage,
            "validation_work_equal_coverage":float(np.mean(list(work_coverage.values()))),
            "validation_halfwidth":{"median":float(np.median(half)),"p95":higher_percentile(half,95)},
            "gates":{k:bool(v) for k,v in gates.items()},"qualified":bool(all(gates.values())),"natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"landmarks.csv").open("w",encoding="utf8",newline="") as h:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
