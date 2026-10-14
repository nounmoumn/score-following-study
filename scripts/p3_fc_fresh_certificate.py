#!/usr/bin/env python3
"""One-shot fresh-cohort audit of selective P3-FB online sign certificates."""
from __future__ import annotations
import hashlib,json,sys,warnings
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_clock_repair_preflight as repair
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.temporal import healthy_before
MANIFEST=ROOT/"results/p3_clock_repair_preflight/manifest.json";OUT=ROOT/"results/p3_fc_fresh_certificate"
RADIUS=0.8532530424720652

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def median_state(probability):return int(np.searchsorted(np.cumsum(probability),.5,side="left"))

def load_case(case,hop):
    import partitura as pt
    perf=ROOT/case["performance"];score_path=ROOT/case["score"];match=ROOT/case["match"]
    for path,key in ((perf,"performance"),(score_path,"score"),(match,"match")):
        if sha(path)!=case["sha256"][key]:raise AssertionError(f"hash changed: {path}")
    score_clock,raw=repair.raw_clock(match);perf_clock=repair.pava(raw)
    if not np.all(np.diff(perf_clock)>0):raise AssertionError("repaired clock is not strict")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");parsed=pt.load_musicxml(str(score_path));array=pt.score.unfold_part_maximal(parsed[0],ignore_leaps=False).note_array()
    score_notes=[(float(r["onset_beat"])*.5,float(r["onset_beat"]+r["duration_beat"])*.5,int(r["pitch"])) for r in array]
    perf_notes=load_notes(perf);perf_duration=max(n[1] for n in perf_notes);score_duration=max(n[1] for n in score_notes)
    score_hop=hop*score_duration/perf_duration
    u=notes_to_chroma_stream(perf_notes,perf_duration,hop);v=notes_to_chroma_stream(score_notes,score_duration,score_hop)
    truth=np.interp(np.arange(len(u))*hop,perf_clock,score_clock,left=score_clock[0],right=score_clock[-1])/score_hop
    return u,v,score_hop,truth,perf_duration

def main():
    manifest=json.loads(MANIFEST.read_text());cases=[r for r in manifest["rows"] if r["eligible"]]
    if len(cases)!=11:raise SystemExit("fresh manifest changed")
    cfg=yaml.safe_load((ROOT/"configs/p3_temporal.yaml").read_text());hop=float(cfg["hop_sec"]);weights=tuple(float(x) for x in cfg["bayes"]["transition"])
    rows=[];path_mismatches=0;normalization=0.
    for number,case in enumerate(cases,1):
        u,v,score_hop,truth,duration=load_case(case,hop);path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights)
        path_mismatches+=int(np.sum(path!=trace.argmax(axis=1)));normalization=max(normalization,float(np.max(np.abs(trace.sum(axis=1)-1))))
        error=np.abs(path-truth)*score_hop;states=np.arange(len(v))
        for time_sec in np.arange(15.,duration-2.1+1e-9,5.):
            end=int(round(time_sec/hop));endpoint=end+20
            if endpoint>=len(trace) or not healthy_before(error,end-1,20,.5):continue
            predicted=trace[end-1].copy()
            for _ in range(21):predicted=dc.apply_prediction(predicted,weights)
            center=median_state(predicted);position_error=abs(center-truth[endpoint])*score_hop;covered=position_error<=RADIUS
            z=fp.projected_action(trace,weights,end-20,end,21)
            predicted_risk=np.minimum(np.abs(states-center)*score_hop,5.);true_risk=np.minimum(np.abs(states-truth[endpoint])*score_hop,5.)
            predicted_help=-float(predicted_risk@z);oracle_help=-float(true_risk@z);bound=RADIUS*float(np.abs(z).sum())
            certified=abs(predicted_help)>bound
            rows.append({"source":case["source"],"unit":case["unit"],"covered":covered,"certified":certified,
                         "position_error":position_error,"predicted_help":predicted_help,"oracle_help":oracle_help,"bound":bound,
                         "bound_violation":abs(predicted_help-oracle_help)-bound,
                         "sign_match":bool(np.sign(predicted_help)==np.sign(oracle_help))})
        print(f"[{number}/11] {case['source']} {case['unit']}")
    tracks={r["unit"] for r in rows};coverage=float(np.mean([r["covered"] for r in rows]));certificate=float(np.mean([r["certified"] for r in rows]))
    source_coverage={s:float(np.mean([r["covered"] for r in rows if r["source"]==s])) for s in ("KRAISLER","Batik")}
    certified_tracks={s:len({r["unit"] for r in rows if r["source"]==s and r["certified"]}) for s in ("KRAISLER","Batik")}
    covered_cert=[r for r in rows if r["covered"] and r["certified"]]
    gates={"coverage_counts":len(tracks)>=9 and len(rows)>=100,"position_coverage_ge_0_90":coverage>=.90,
           "each_source_coverage_ge_0_85":min(source_coverage.values())>=.85,"certificate_share_ge_0_20":certificate>=.20,
           "two_certified_tracks_each_source":min(certified_tracks.values())>=2,
           "zero_covered_bound_violations":all(r["bound_violation"]<=1e-12 for r in rows if r["covered"]),
           "zero_certified_sign_mismatches":all(r["sign_match"] for r in covered_cert),
           "path_and_probability_audit":path_mismatches==0 and normalization<=1e-12}
    result={"hypothesis":"selective online projected-help sign certificate","counts":{"manifest_tracks":len(cases),"healthy_tracks":len(tracks),"landmarks":len(rows),"covered_certificates":len(covered_cert)},
            "radius_sec":RADIUS,"position_coverage":coverage,"source_coverage":source_coverage,"certificate_share":certificate,
            "certified_tracks":certified_tracks,"max_covered_bound_violation":max(r["bound_violation"] for r in rows if r["covered"]),
            "gates":{k:bool(v) for k,v in gates.items()},"qualified":bool(all(gates.values())),"natural_temporal_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
