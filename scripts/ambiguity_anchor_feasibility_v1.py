"""Frozen oracle-anchor utility pilot on previously opened performances."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.anchor_feasibility import smooth, candidate_features, choose
from score_following.confidence_audit import blocked_maxsim
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),encoding="utf-8")


def main():
    out = ROOT / "results/ambiguity_anchor_feasibility_v1"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    manifest = ROOT / "output/score_priority_plan/cohort_manifest.csv"
    annotation_file = ROOT / "asap_dataset/asap_annotations.json"
    with manifest.open(encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f) if r["role"] == "primary_opened"]
    assert len(cases) == 40
    ann = json.loads(annotation_file.read_text(encoding="utf-8"))
    files = [Path(__file__), ROOT / "AMBIGUITY_ANCHOR_FEASIBILITY_V1_PROTOCOL.md", manifest, annotation_file,
             ROOT / "src/score_following/anchor_feasibility.py", ROOT / "tests/test_anchor_feasibility.py"]
    files += [ROOT / "src/score_following" / (s+".py") for s in
              ("features","confidence_audit","p3_extension","observation_evidence","fixed_lag_smoothing")]
    files += sorted({ROOT / "asap_dataset" / r[k] for r in cases for k in ("score","performance")})
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out.mkdir()
    write(out / "freeze.json", {"scope":"opened development, offline oracle anchors", "files":hashes})
    write(out / "state.json", {"status":"RUNNING"})
    candidates = np.arange(15,86,dtype=float)
    query_idx = (candidates*10).astype(int)-1
    policies = [(p,0) for p in ("uniform","uncertainty","ambiguity","joint","sham_joint")]
    policies += [("random",s) for s in range(3)]
    records, cache = [], {}
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for number, case in enumerate(cases,1):
            case_start = time.perf_counter()
            pn = load_notes(ROOT / "asap_dataset" / case["performance"])
            if case["score"] not in cache:
                sn = load_notes(ROOT / "asap_dataset" / case["score"])
                v = notes_to_chroma_stream(sn,max(n[1] for n in sn),.1)
                a = blocked_maxsim(v,20,20,256)[0]
                cache[case["score"]] = v,a
            v,a = cache[case["score"]]
            frames = int(np.floor(min(90.,max(n[1] for n in pn))/.1))
            if frames < 850:
                raise ValueError("prefix too short for frozen candidates")
            cost = cosine_cost(notes_to_chroma_stream(pn,frames*.1,.1)[:frames],v)
            gt = ann[case["performance"]]
            times,truth = np.array(gt["performance_beats"]), np.array(gt["midi_score_beats"])
            if np.any(np.diff(times)<=0) or np.any(np.diff(truth)<=0):
                raise ValueError("invalid clock")
            if times[0] > 15 or times[-1] < 85:
                raise ValueError("query outside annotations")
            keep = (times>=15)&(times<=85)&(abs(times[:,None]-candidates).min(1)>.11)
            eval_times, eval_truth = times[keep],truth[keep]
            idx = np.searchsorted((np.arange(frames)+1)*.1,eval_times,side="right")-1
            base = smooth(cost)
            arank, urank = candidate_features(base,a,query_idx)
            base_path = (base.argmax(1)+1)*.1
            base_y = abs(base_path[idx]-eval_truth)<=.5+1e-12
            data = {"case_id":case["case_id"],"work_family":case["work_family"],
                    "evaluation_rows":len(idx),"baseline_accuracy":float(base_y.mean()),"conditions":{}}
            del base
            for budget in (1,4):
                for policy,seed in policies:
                    selected = choose(candidates,arank,urank,budget,policy,seed)
                    qt = candidates[selected]
                    supplied = np.interp(qt,times,truth)
                    state = np.rint(supplied/.1).astype(int)-1
                    if np.any(state<0) or np.any(state>=len(v)):
                        raise ValueError("invalid anchor state")
                    corrected = smooth(cost,dict(zip(query_idx[selected],state)))
                    path = (corrected.argmax(1)+1)*.1
                    y = abs(path[idx]-eval_truth)<=.5+1e-12
                    key = f"{policy}_{seed}_k{budget}"
                    data["conditions"][key] = {"accuracy":float(y.mean()),
                        "fixed_fraction":float((y & ~base_y).mean()),
                        "broken_fraction":float((~y & base_y).mean()),
                        "query_times":qt.tolist(),"supplied_truth":supplied.tolist(),
                        "anchor_states":state.tolist()}
                    del corrected
            records.append(data)
            write(out / (case["case_id"]+".json"),data)
            print(f"[{number}/40] {case['case_id']} base={data['baseline_accuracy']:.3f} "
                  f"joint4={data['conditions']['joint_0_k4']['accuracy']:.3f} "
                  f"{time.perf_counter()-case_start:.1f}s",flush=True)
    keys = list(records[0]["conditions"])
    families = sorted({r["work_family"] for r in records})
    grouped = {key:np.array([np.mean([r["conditions"][key]["accuracy"] for r in records if r["work_family"]==w])
                            for w in families]) for key in keys}
    baseline = np.array([np.mean([r["baseline_accuracy"] for r in records if r["work_family"]==w]) for w in families])
    summary = {k:{"accuracy":float(v.mean()),"gain_vs_no_anchor":float((v-baseline).mean())} for k,v in grouped.items()}
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0,len(families),(10000,len(families)))
    comparisons = {}
    for left,right in (("joint_0_k4","uncertainty_0_k4"),("ambiguity_0_k4","uniform_0_k4"),
                       ("joint_0_k4","sham_joint_0_k4")):
        diff = grouped[left]-grouped[right]
        comparisons[left+" minus "+right] = {"mean":float(diff.mean()),
            "ci95":np.quantile(diff[sample].mean(1),[.025,.975]).tolist(),
            "positive_works":int((diff>0).sum()),"negative_works":int((diff<0).sum())}
    main = comparisons["joint_0_k4 minus uncertainty_0_k4"]
    gate = main["mean"]>=.01 and main["ci95"][0]>0 and comparisons["ambiguity_0_k4 minus uniform_0_k4"]["mean"]>0 and comparisons["joint_0_k4 minus sham_joint_0_k4"]["mean"]>0
    drift = [p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not drift
    result = {"scope":"opened development offline oracle pilot", "cases":len(records),"works":len(families),
        "evaluation_rows":sum(r["evaluation_rows"] for r in records),"baseline_accuracy":float(baseline.mean()),
        "metrics":summary,"comparisons":comparisons,"development_advance_gate":bool(gate),
        "hash_drift":drift,"wall_sec":time.perf_counter()-started}
    write(out / "result.json",result)
    write(out / "state.json",{"status":"COMPLETE","advance_gate":bool(gate)})
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
