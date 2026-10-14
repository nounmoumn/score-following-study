"""Frozen causal replay and retrospective diagnostic; no tuning or recommendation."""
import csv, hashlib, json, sys, time, platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
from score_following.p3_extension import cosine_cost,oltw_from_cost,bayes_from_cost
from score_following.fixed_lag_smoothing import smoothed_trace

def read(p): return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=ROOT/'results/online_ambiguity_diagnosis_v2'
    if out.exists(): raise SystemExit('Refuse overwrite')
    manifest=ROOT/'output/score_priority_plan/cohort_manifest.csv'
    cases=[r for r in read(manifest) if r['role']=='primary_opened']
    obsfile=ROOT/'results/p2_context_utility_feasibility/observations.csv'
    obs=read(obsfile)
    gtfile=ROOT/'results/confidence_smoothing_v1/delay_free_beats.csv'
    gt={(r['case_id'],round(float(r['performance_time_sec']),6)):r for r in read(gtfile) if int(r['lag'])==0}
    oldhash=json.loads((ROOT/'results/ambiguity_contrast_v1/freeze.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in oldhash.items())
    sources=[Path(__file__),ROOT/'ONLINE_AMBIGUITY_DIAGNOSIS_PROTOCOL.md',ROOT/'scripts/analyze_online_ambiguity_diagnosis.py',manifest,obsfile,gtfile]
    sources+=list((ROOT/'src/score_following').glob('*.py'))
    sources+=[ROOT/'asap_dataset'/r[k] for r in cases for k in ['score','performance']]
    frozen={str(p.relative_to(ROOT)):sha(p) for p in sources}
    out.mkdir();write(out/'freeze.json',frozen)
    write(out/'state.json',{'status':'RUNNING','python':sys.version,'platform':platform.platform()})
    records=[];cache={}; audits=[]
    with threadpool_limits(limits=2):
      for c in cases:
        tic=time.perf_counter()
        samples=[r for r in obs if r['case_id']==c['case_id']]
        if c['score'] not in cache:
            sn=load_notes(ROOT/'asap_dataset'/c['score']);duration=max(n[1] for n in sn)
            sr=piano_roll(sn,duration);sc=notes_to_chroma_stream(sn,duration,.1)
            cache[c['score']]=(sn,sr,sc)
        sn,sr,sc=cache[c['score']]
        pn=load_notes(ROOT/'asap_dataset'/c['performance'])
        frames=int(np.floor(min(90.,max(n[1] for n in pn))/.1))
        pr=piano_roll(pn,frames*.1)[:frames];pc=notes_to_chroma_stream(pn,frames*.1,.1)[:frames]
        pred={}
        for name,perf,score in [('chroma',pc,sc),('pitch',normalize(pr),normalize(sr))]:
            cost=cosine_cost(perf,score)
            pred['bayes_'+name]=bayes_from_cost(cost)
            pred['dtw_'+name]=oltw_from_cost(cost,60)
            pred['lag_'+name]=smoothed_trace(cost,20)['map_index'].astype(int)
            cut=min(420,frames-21)
            np.testing.assert_array_equal(bayes_from_cost(cost[:cut+1]),pred['bayes_'+name][:cut+1])
            np.testing.assert_array_equal(oltw_from_cost(cost[:cut+1],60)[:cut],pred['dtw_'+name][:cut])
            np.testing.assert_array_equal(smoothed_trace(cost[:cut+1],20)['map_index'],pred['lag_'+name][:cut+1-20])
            del cost
        # A sustained note is clipped at the observation boundary; its future offset is unknown.
        boundary=42.
        prefix=[(s,min(e,boundary),p) for s,e,p in pn if s<boundary]
        np.testing.assert_allclose(piano_roll(prefix,boundary)[:420],pr[:420],atol=1e-13)
        np.testing.assert_allclose(notes_to_chroma_stream(prefix,boundary,.1)[:420],pc[:420],atol=1e-13)
        truths=np.array([float(gt[(c['case_id'],round(float(r['time']),6))]['truth_score_sec']) for r in samples])
        times=np.array([float(r['time']) for r in samples])
        tempo=np.gradient(truths,times)
        rows=[]
        for k,r in enumerate(samples):
            t=times[k];truth=truths[k];f=int(np.searchsorted((np.arange(frames)+1)*.1,t,side='right')-1)
            assert 0<=f and f+20<frames
            assert r['work_family']==c['work_family']
            i=int(round(truth/.1))-1;start=i*.1;end=start+2
            active=[n for n in sn if n[0]<end and n[1]>start]
            pitches=[n[2] for n in active]
            controls=[sum(start<=n[0]<end for n in sn)/2.,float(max(pitches)-min(pitches)) if pitches else 0.,
                      float(sr[i:i+20].sum()/(.1*20)),float(tempo[k])]
            positions={key:float((v[f]+1)*.1) for key,v in pred.items()}
            errors={key:abs(value-truth) for key,value in positions.items()}
            correct={key:int(value<=.5+1e-12) for key,value in errors.items()}
            assert correct['bayes_chroma']==int(r['base_correct'])
            assert correct['lag_chroma']-correct['bayes_chroma']==int(r['gain_20'])
            aval=float(r['A_true'])
            if not np.isfinite(aval):
                assert i == -1 and truth == 0., (c['case_id'],truth,i)
                aval=None;controls=None
            row=dict(time=float(t),truth=float(truth),A=aval,controls=controls,
                     frame=f,available=(f+1)*.1,lag_available=(f+21)*.1,positions=positions,errors=errors,correct=correct)
            if aval is not None: assert np.isfinite([aval]+controls).all()
            rows.append(row)
        record=dict(case=c['case_id'],work=c['work_family'],composer=c['composer'],rows=rows)
        write(out/(c['case_id']+'.json'),record);records.append(record)
        audits.append(dict(case=c['case_id'],prefix_equivalence=True,feature_prefix=True,old_chroma_equivalence=True,seconds=time.perf_counter()-tic))
        print(c['case_id'],len(rows),round(audits[-1]['seconds'],2),flush=True)
    assert len(records)==40 and sum(len(c['rows']) for c in records)==4889
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    write(out/'audit.json',{'status':'PASS','rows':4889,'cases':audits,'hash_drift':0})
    write(out/'state.json',{'status':'INFERENCE_COMPLETE'})

if __name__=='__main__':main()
