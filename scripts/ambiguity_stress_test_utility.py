"""Frozen external-work replication of ambiguity's incremental diagnostic utility."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.confidence_audit import blocked_maxsim
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
from score_following.p3_extension import cosine_cost,bayes_from_cost
from score_following.tracker_candidate_diagnosis import trace

OUT=ROOT/'results/ambiguity_stress_test_utility'
CAP=5.;SEED=20260923

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def weights(cases):
 works=sorted({c['work'] for c in cases});out=[]
 for c in cases:
  nc=sum(x['work']==c['work'] for x in cases);out.extend([1/(len(works)*nc*len(c['rows']))]*len(c['rows']))
 return np.array(out),works
def ridge_predict(xtr,ytr,wtr,xte,alpha=1.):
 mu=np.average(xtr,axis=0,weights=wtr);sd=np.sqrt(np.average((xtr-mu)**2,axis=0,weights=wtr));sd=np.maximum(sd,1e-12)
 z=np.column_stack([np.ones(len(xtr)),(xtr-mu)/sd]);zt=np.column_stack([np.ones(len(xte)),(xte-mu)/sd])
 reg=np.eye(z.shape[1]);reg[0,0]=0
 b=np.linalg.solve(z.T@(wtr[:,None]*z)+alpha*reg,z.T@(wtr*ytr))
 return z@b,zt@b,dict(mean=mu.tolist(),sd=sd.tolist(),coefficients=b.tolist())
def bootstrap_ci(values):
 values=np.asarray(values,float);rng=np.random.default_rng(SEED)
 draws=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
 return [float(x) for x in np.quantile(draws,[.025,.975])]
def work_means(values,cases,works):
 values=np.asarray(values);answer=[];start=0
 for work in works:
  per_case=[]
  for c in cases:
   n=len(c['rows']);part=values[start:start+n]
   if c['work']==work:per_case.append(float(part.mean()))
   start+=n
  answer.append(float(np.mean(per_case)))
  start=0
 return np.array(answer)
def load_development():
 cases=[]
 for p in sorted((ROOT/'results/tracker_candidate_diagnosis').glob('c[0-9][0-9][0-9].json')):
  c=json.loads(p.read_text());rows=[]
  for r in c['rows']:
   if r['A'] is None:continue
   rows.append(dict(A=float(r['A']),state=[float(v) for v in r['candidate']['base']],
                    chroma=min(float(r['errors']['bayes_chroma']),CAP),pitch=min(float(r['errors']['bayes_pitch']),CAP)))
  cases.append(dict(case=c['case'],work=c['work'],rows=rows))
 return cases
def build_evaluation(meta,annotations):
 cases=[];score_cache={};prefix_checks=0
 with threadpool_limits(limits=2):
  for num,m in enumerate(meta,1):
   ann=annotations[m['performance']]
   if not ann.get('score_and_performance_aligned',False):raise ValueError('unaligned '+m['case_id'])
   if m['score'] not in score_cache:
    sn=load_notes(ROOT/'asap_dataset'/m['score']);duration=max(v[1] for v in sn)
    sc=notes_to_chroma_stream(sn,duration,.1);sp=normalize(piano_roll(sn,duration));A=blocked_maxsim(sc,20,20,256)[0]
    score_cache[m['score']]=(sc,sp,A)
   sc,sp,A=score_cache[m['score']]
   pn=load_notes(ROOT/'asap_dataset'/m['performance']);nf=int(np.floor(min(90.,max(v[1] for v in pn))/.1))
   pc=notes_to_chroma_stream(pn,nf*.1,.1)[:nf];pp=normalize(piano_roll(pn,nf*.1)[:nf])
   cc=cosine_cost(pc,sc);cp=cosine_cost(pp,sp)
   mc=bayes_from_cost(cc);mp=bayes_from_cost(cp)
   perf=np.asarray(ann['performance_beats'],float);truth=np.asarray(ann['midi_score_beats'],float)
   mask=(perf>=2.1)&(perf<=(nf-20)*.1);perf=perf[mask];truth=truth[mask]
   if len(perf)<20:raise ValueError('too few beats '+m['case_id'])
   frames=np.searchsorted((np.arange(nf)+1)*.1,perf,side='right')-1
   maps,features=trace(cc,sc,sp,frames.tolist())
   np.testing.assert_array_equal(mc,maps)
   cut=min(420,nf-1);check_frames=[int(f) for f in frames if f<=cut]
   cm,cf=trace(cc[:cut+1],sc,sp,check_frames)
   np.testing.assert_array_equal(mc[:cut+1],cm)
   for f in check_frames:
    np.testing.assert_allclose(features[f]['base'],cf[f]['base'],atol=1e-14,rtol=0)
   prefix_checks+=1;rows=[]
   for t,gt,f in zip(perf,truth,frames):
    idx=int(round(gt/.1))-1
    if idx<0 or idx>=len(A) or not np.isfinite(A[idx]):continue
    ec=abs((mc[f]+1)*.1-gt);ep=abs((mp[f]+1)*.1-gt);base=[float(v) for v in features[int(f)]['base']]
    if not np.isfinite([A[idx],ec,ep]+base).all():raise ValueError('nonfinite '+m['case_id'])
    rows.append(dict(time=float(t),truth=float(gt),A=float(A[idx]),state=base,chroma=min(float(ec),CAP),pitch=min(float(ep),CAP)))
   if len(rows)<20:raise ValueError('too few valid rows '+m['case_id'])
   cases.append(dict(case=m['case_id'],work=m['work_family'],rows=rows))
   print(f"[{num}/{len(meta)}] {m['case_id']} rows={len(rows)}",flush=True)
 return cases,prefix_checks
def analyze(dev,eva):
 wd,devworks=weights(dev);we,works=weights(eva)
 dr=[r for c in dev for r in c['rows']];er=[r for c in eva for r in c['rows']]
 yd=np.array([r['chroma']-r['pitch'] for r in dr]);ye=np.array([r['chroma']-r['pitch'] for r in er])
 bd=np.array([r['state'] for r in dr]);be=np.array([r['state'] for r in er]);ad=np.array([r['A'] for r in dr]);ae=np.array([r['A'] for r in er])
 _,pb,fitb=ridge_predict(bd,yd,wd*len(wd),be);_,pa,fita=ridge_predict(np.column_stack([bd,ad]),yd,wd*len(wd),np.column_stack([be,ae]))
 mseb=work_means((ye-pb)**2,eva,works);msea=work_means((ye-pa)**2,eva,works);improve=mseb-msea
 actual=work_means(ye,eva,works);predb=work_means(pb,eva,works);preda=work_means(pa,eva,works)
 lc=work_means(np.array([r['chroma'] for r in er]),eva,works);lp=work_means(np.array([r['pitch'] for r in er]),eva,works);oracle=np.minimum(lc,lp)
 global_pitch=bool(np.average(yd,weights=wd)>=0)
 losses={'global':np.where(global_pitch,lp,lc),'state':np.where(predb>=0,lp,lc),'state_A':np.where(preda>=0,lp,lc)}
 regrets={k:v-oracle for k,v in losses.items()};d_state=regrets['state']-regrets['state_A'];d_global=regrets['global']-regrets['state_A']
 gate1=float(improve.mean())>0 and bootstrap_ci(improve)[0]>0
 gate2=int((improve>0).sum())>=int(np.ceil(.6*len(works)))
 gate3=float(d_state.mean())>0 and bootstrap_ci(d_state)[0]>0 and float(d_global.mean())>0 and bootstrap_ci(d_global)[0]>0
 return {'scope':'outcome-exposed external-work replication, not fresh confirmation','development':{'cases':len(dev),'works':len(devworks),'rows':len(dr)},
  'evaluation':{'cases':len(eva),'works':len(works),'rows':len(er)},'target':'min(error_chroma,5)-min(error_pitch,5)',
  'primary_mse':{'state':float(mseb.mean()),'state_A':float(msea.mean()),'improvement':float(improve.mean()),'bootstrap95':bootstrap_ci(improve),'positive_works':int((improve>0).sum()),'per_work':[dict(work=w,state=float(mseb[i]),state_A=float(msea[i]),improvement=float(improve[i])) for i,w in enumerate(works)]},
  'selection_regret':{k:{'mean':float(v.mean()),'bootstrap95_mean':bootstrap_ci(v)} for k,v in regrets.items()}|{'state_minus_state_A':{'mean':float(d_state.mean()),'bootstrap95':bootstrap_ci(d_state)},'global_minus_state_A':{'mean':float(d_global.mean()),'bootstrap95':bootstrap_ci(d_global)},'global_choice':'pitch' if global_pitch else 'chroma','per_work':[dict(work=w,actual_gain=float(actual[i]),state_prediction=float(predb[i]),state_A_prediction=float(preda[i]),oracle_loss=float(oracle[i]),global_regret=float(regrets['global'][i]),state_regret=float(regrets['state'][i]),state_A_regret=float(regrets['state_A'][i])) for i,w in enumerate(works)]},
  'gates':{'mse_mean_and_ci':gate1,'mse_60pct_works':gate2,'selection_better_than_state_and_global_with_ci':gate3,'development_validation_pass':bool(gate1 and gate2 and gate3)},'fits':{'state':fitb,'state_A':fita}}
def main():
 if OUT.exists():raise SystemExit('Refuse overwrite')
 manifest=ROOT/'output/score_priority_plan/cohort_manifest.csv';protocol=ROOT/'AMBIGUITY_STRESS_TEST_UTILITY_PROTOCOL.md'
 allmeta=read_csv(manifest);devworks={r['work_family'] for r in allmeta if r['role']=='primary_opened'}
 meta=[r for r in allmeta if r['role']=='replication_opened_nonoverlap']
 assert len(meta)==36 and not (devworks&{r['work_family'] for r in meta})
 annotation_file=ROOT/'asap_dataset/asap_annotations.json';annotations=json.loads(annotation_file.read_text())
 files=[Path(__file__),protocol,manifest,annotation_file,ROOT/'src/score_following/confidence_audit.py',ROOT/'src/score_following/tracker_candidate_diagnosis.py',ROOT/'src/score_following/p3_extension.py',ROOT/'src/score_following/features.py']
 files+=sorted((ROOT/'results/tracker_candidate_diagnosis').glob('c[0-9][0-9][0-9].json'))
 files+=[ROOT/'asap_dataset'/r[k] for r in meta for k in ('score','performance')]
 frozen={str(p.relative_to(ROOT)):sha(p) for p in files};OUT.mkdir();write(OUT/'freeze.json',frozen);write(OUT/'state.json',{'status':'RUNNING'})
 dev=load_development();eva,prefix=build_evaluation(meta,annotations);result=analyze(dev,eva)
 assert all(sha(ROOT/p)==h for p,h in frozen.items())
 for c in eva:write(OUT/(c['case']+'.json'),c)
 write(OUT/'result.json',result);write(OUT/'audit.json',{'status':'PASS','prefix_cases':prefix,'hash_drift':0,'development_work_overlap':0});write(OUT/'state.json',{'status':'COMPLETE'})
 print(json.dumps({**result,'fits':'omitted','primary_mse':{k:v for k,v in result['primary_mse'].items() if k!='per_work'},'selection_regret':{k:v for k,v in result['selection_regret'].items() if k!='per_work'}},indent=2))
if __name__=='__main__':main()
