"""Two-stage frozen ambiguity-map pipeline. Usage: script development|lockbox"""
import csv,hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.confidence_audit import blocked_maxsim
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
from score_following.p3_extension import cosine_cost,bayes_from_cost
from score_following.tracker_candidate_diagnosis import trace
CAP=5.;SEED=20260923;SYSTEMS=('chroma','pitch','fused');PAIRS=(('chroma','pitch'),('pitch','fused'),('chroma','fused'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def readcsv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def weights(cases):
 works=sorted({c['work'] for c in cases});out=[]
 for c in cases:
  nc=sum(x['work']==c['work'] for x in cases);out.extend([1/(len(works)*nc*len(c['rows']))]*len(c['rows']))
 return np.array(out),works
def wmean(values,cases,works):
 values=np.asarray(values);answer=[]
 for work in works:
  start=0;per=[]
  for c in cases:
   n=len(c['rows'])
   if c['work']==work:per.append(float(values[start:start+n].mean()))
   start+=n
  answer.append(float(np.mean(per)))
 return np.array(answer)
def fit(x,y,w):
 mu=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w));sd=np.maximum(sd,1e-12);z=np.column_stack([np.ones(len(x)),(x-mu)/sd]);reg=np.eye(z.shape[1]);reg[0,0]=0;b=np.linalg.solve(z.T@(w[:,None]*z)+reg,z.T@(w*y));return {'mean':mu.tolist(),'sd':sd.tolist(),'coef':b.tolist()}
def predict(x,m):
 return np.asarray(m['coef'])[0]+((x-np.asarray(m['mean']))/np.asarray(m['sd']))@np.asarray(m['coef'])[1:]
def ci(v):
 v=np.asarray(v);rng=np.random.default_rng(SEED);d=v[rng.integers(0,len(v),(10000,len(v)))].mean(1);return np.quantile(d,[.025,.975]).tolist()
def infer(rows,annotations,out):
 cases=[];cache={}
 with threadpool_limits(limits=2):
  for number,m in enumerate(rows,1):
   ann=annotations[m['performance']];assert ann.get('score_and_performance_aligned')
   if m['score'] not in cache:
    sn=load_notes(ROOT/'asap_dataset'/m['score']);duration=max(x[1] for x in sn);sc=notes_to_chroma_stream(sn,duration,.1);sp=normalize(piano_roll(sn,duration));cache[m['score']]=(sc,sp,blocked_maxsim(sc,20,20,256)[0])
   sc,sp,A=cache[m['score']];pn=load_notes(ROOT/'asap_dataset'/m['performance']);nf=int(np.floor(min(90.,max(x[1] for x in pn))/.1));pc=notes_to_chroma_stream(pn,nf*.1,.1)[:nf];pp=normalize(piano_roll(pn,nf*.1)[:nf]);cost={'chroma':cosine_cost(pc,sc),'pitch':cosine_cost(pp,sp)};cost['fused']=.5*(cost['chroma']+cost['pitch'])
   perf=np.asarray(ann['performance_beats']);truth=np.asarray(ann['midi_score_beats']);mask=(perf>=.1)&(perf<=nf*.1);perf=perf[mask];truth=truth[mask];frames=np.searchsorted((np.arange(nf)+1)*.1,perf,side='right')-1
   maps={};features={}
   for s in SYSTEMS:
    maps[s],features[s]=trace(cost[s],sc,sp,frames.tolist());np.testing.assert_array_equal(maps[s],bayes_from_cost(cost[s]));cut=min(420,nf-1);fs=[int(f) for f in frames if f<=cut];cm,cf=trace(cost[s][:cut+1],sc,sp,fs);np.testing.assert_array_equal(maps[s][:cut+1],cm)
    for f in fs:np.testing.assert_allclose(features[s][f]['base'],cf[f]['base'],atol=1e-14,rtol=0)
   rr=[]
   for t,gt,f in zip(perf,truth,frames):
    ix=int(round(gt/.1))-1
    if ix<0 or ix>=len(A) or not np.isfinite(A[ix]):continue
    pos={s:float((maps[s][f]+1)*.1) for s in SYSTEMS};err={s:float(min(abs(pos[s]-gt),CAP)) for s in SYSTEMS};state={s:[float(v) for v in features[s][int(f)]['base']] for s in SYSTEMS}
    if not np.isfinite([A[ix]]+list(err.values())+sum(state.values(),[])).all():raise ValueError('nonfinite')
    rr.append(dict(time=float(t),truth=float(gt),A=float(A[ix]),position=pos,loss=err,state=state))
   if len(rr)<20:raise ValueError('too few rows '+m['case_id'])
   c=dict(case=m['case_id'],work=m['work_family'],composer=m['composer'],rows=rr);cases.append(c);write(out/(m['case_id']+'.json'),c);print(f'[{number}/{len(rows)}] {m["case_id"]} rows={len(rr)}',flush=True)
 return cases
def screen(cases):
 w,works=weights(cases);r=[x for c in cases for x in c['rows']];summary={};workloss={};workacc={}
 for s in SYSTEMS:
  loss=np.array([x['loss'][s] for x in r]);acc=np.array([x['loss'][s]<=.5+1e-12 for x in r]);workloss[s]=wmean(loss,cases,works);workacc[s]=wmean(acc,cases,works);summary[s]={'accuracy':float(workacc[s].mean()),'capped_loss':float(workloss[s].mean())}
 checks=[];chosen=None
 for a,b in PAIRS:
  wins_a=int((workloss[a]<workloss[b]).sum());wins_b=int((workloss[b]<workloss[a]).sum());disagree=float(np.average([abs(x['position'][a]-x['position'][b])>.5+1e-12 for x in r],weights=w));ratio=max(summary[a]['capped_loss'],summary[b]['capped_loss'])/max(min(summary[a]['capped_loss'],summary[b]['capped_loss']),1e-12)
  ok=abs(summary[a]['accuracy']-summary[b]['accuracy'])<=.03 and ratio<=1.15 and wins_a>=int(np.ceil(.2*len(works))) and wins_b>=int(np.ceil(.2*len(works))) and disagree>=.05
  checks.append({'pair':[a,b],'accuracy_gap':abs(summary[a]['accuracy']-summary[b]['accuracy']),'loss_ratio':ratio,'work_wins':[wins_a,wins_b],'map_disagreement':disagree,'pass':bool(ok)})
  if ok and chosen is None:chosen=(a,b)
 return summary,checks,chosen
def development(cases):
 summary,checks,pair=screen(cases);result={'systems':summary,'pair_checks':checks,'selected_pair':list(pair) if pair else None,'lockbox_authorized':pair is not None}
 if not pair:return result,None
 a,b=pair;w,works=weights(cases);rows=[r for c in cases for r in c['rows']];y=np.array([r['loss'][a]-r['loss'][b] for r in rows]);state=np.array([r['state'][a] for r in rows]);A=np.array([r['A'] for r in rows]);model={'pair':[a,b],'state':fit(state,y,w*len(w)),'state_A':fit(np.column_stack([state,A]),y,w*len(w)),'global_choice':b if np.average(y,weights=w)>=0 else a,'target':'loss(first)-loss(second)','cap':CAP};result['development_target_mean']=float(np.average(y,weights=w));return result,model
def lockbox(cases,model):
 a,b=model['pair'];w,works=weights(cases);rows=[r for c in cases for r in c['rows']];y=np.array([r['loss'][a]-r['loss'][b] for r in rows]);state=np.array([r['state'][a] for r in rows]);A=np.array([r['A'] for r in rows]);pb=predict(state,model['state']);pa=predict(np.column_stack([state,A]),model['state_A']);mb=wmean((y-pb)**2,cases,works);ma=wmean((y-pa)**2,cases,works);imp=mb-ma;actual=wmean(y,cases,works);pmb=wmean(pb,cases,works);pma=wmean(pa,cases,works);la=wmean(np.array([r['loss'][a] for r in rows]),cases,works);lb=wmean(np.array([r['loss'][b] for r in rows]),cases,works);oracle=np.minimum(la,lb);global_loss=lb if model['global_choice']==b else la;rg=global_loss-oracle;rs=np.where(pmb>=0,lb,la)-oracle;ra=np.where(pma>=0,lb,la)-oracle;ds=rs-ra;dg=rg-ra;g1=imp.mean()>0 and ci(imp)[0]>0;g2=(imp>0).sum()>=int(np.ceil(.6*len(works)));g3=ds.mean()>0 and ci(ds)[0]>0 and dg.mean()>0 and ci(dg)[0]>0
 return {'pair':[a,b],'cases':len(cases),'works':len(works),'rows':len(rows),'mse':{'state':float(mb.mean()),'state_A':float(ma.mean()),'improvement':float(imp.mean()),'bootstrap95':ci(imp),'positive_works':int((imp>0).sum())},'regret':{'global':float(rg.mean()),'state':float(rs.mean()),'state_A':float(ra.mean()),'state_minus_state_A':float(ds.mean()),'state_minus_state_A_ci':ci(ds),'global_minus_state_A':float(dg.mean()),'global_minus_state_A_ci':ci(dg)},'gates':{'mse_ci':bool(g1),'positive_60pct':bool(g2),'decision':bool(g3),'pass':bool(g1 and g2 and g3)},'per_work':[dict(work=x,actual=float(actual[i]),state_prediction=float(pmb[i]),state_A_prediction=float(pma[i]),mse_improvement=float(imp[i]),global_regret=float(rg[i]),state_regret=float(rs[i]),state_A_regret=float(ra[i])) for i,x in enumerate(works)]}
def main():
 if len(sys.argv)!=2 or sys.argv[1] not in ('development','lockbox'):raise SystemExit('usage: development|lockbox')
 role=sys.argv[1];out=ROOT/f'results/ambiguity_map_fresh_{role if role=="development" else "lockbox"}'
 if out.exists():raise SystemExit('Refuse overwrite')
 manifest=ROOT/'output/ambiguity_map_fresh_cohort/manifest.csv';protocol=ROOT/'AMBIGUITY_MAP_FRESH_PIPELINE_PROTOCOL.md';meta=[r for r in readcsv(manifest) if r['role']==role];assert len(meta)==8
 model_file=ROOT/'results/ambiguity_map_fresh_development/model.json'
 if role=='lockbox':
  state=json.loads((ROOT/'results/ambiguity_map_fresh_development/state.json').read_text());assert state['lockbox_authorized'];model=json.loads(model_file.read_text())
 else:model=None
 annotation=ROOT/'asap_dataset/asap_annotations.json';files=[Path(__file__),protocol,manifest,annotation,ROOT/'src/score_following/confidence_audit.py',ROOT/'src/score_following/tracker_candidate_diagnosis.py',ROOT/'src/score_following/p3_extension.py',ROOT/'src/score_following/features.py']+[ROOT/'asap_dataset'/r[k] for r in meta for k in ('score','performance')]
 if model_file.exists():files.append(model_file)
 freeze={str(p.relative_to(ROOT)):sha(p) for p in files};out.mkdir();write(out/'freeze.json',freeze);write(out/'state.json',{'status':'RUNNING'})
 cases=infer(meta,json.loads(annotation.read_text()),out)
 if role=='development':result,model=development(cases);write(out/'result.json',result)
 else:result=lockbox(cases,model);write(out/'result.json',result)
 assert all(sha(ROOT/p)==h for p,h in freeze.items())
 if role=='development' and model is not None:write(out/'model.json',model)
 write(out/'state.json',{'status':'COMPLETE','lockbox_authorized':bool(result.get('lockbox_authorized',False)) if role=='development' else False});print(json.dumps(result,indent=2))
if __name__=='__main__':main()
