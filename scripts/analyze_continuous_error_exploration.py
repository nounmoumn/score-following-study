import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/continuous_error_exploration'
SRC=ROOT/'results/online_ambiguity_diagnosis_v2'
def main():
 if (OUT/'result.json').exists():raise SystemExit('Refuse overwrite')
 OUT.mkdir(exist_ok=True)
 files=[Path(__file__),ROOT/'CONTINUOUS_ERROR_EXPLORATION_PROTOCOL.md']+list(SRC.glob('c[0-9][0-9][0-9].json'))
 h={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (OUT/'freeze.json').write_text(json.dumps(h,indent=2))
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 cases=[json.loads(p.read_text()) for p in sorted(SRC.glob('c[0-9][0-9][0-9].json'))]
 cases=[{**c,'rows':[r for r in c['rows'] if r['A'] is not None]} for c in cases]
 works=sorted({c['work'] for c in cases});W=len(works);rows=[r for c in cases for r in c['rows']]
 ids=np.concatenate([np.full(len(c['rows']),works.index(c['work'])) for c in cases])
 counts={w:sum(c['work']==w for c in cases) for w in works}
 weight=np.concatenate([np.full(len(c['rows']),1/(W*counts[c['work']]*len(c['rows']))) for c in cases])
 X=np.array([[r['A']]+r['controls'] for r in rows])
 samples=np.random.default_rng(20260923).integers(0,W,(5000,W))
 def ci(v):
  v=np.asarray(v);return dict(mean=float(v.mean()),bootstrap95=np.quantile(v[samples].mean(1),[.025,.975]).tolist(),positive_works=int((v>0).sum()),negative_works=int((v<0).sum()))
 result={'scope':'adaptive opened-data exploration, not confirmation','rows':len(rows),'works':W,'comparisons':{},'rival_error_coverage':{}}
 for target in ['bayes','dtw','context_interaction']:
  def val(r):
   k=r['errors']
   return k[target+'_chroma']-k[target+'_pitch'] if target!='context_interaction' else k['lag_chroma']-k['lag_pitch']-k['bayes_chroma']+k['bayes_pitch']
  y=np.array([val(r) for r in rows])
  for feature,col in [('A',0)]:
   errs={False:[],True:[]};slopes={w:[] for w in works};start=0
   for c in cases:
    n=len(c['rows']);a=X[start:start+n,col];a=a-a.mean();sd=a.std()
    slopes[c['work']].append(float(np.mean(a/sd*y[start:start+n])) if sd>0 else 0.);start+=n
   for i in range(W):
    tr=ids!=i;te=~tr;tw=weight[tr]/weight[tr].sum()*tr.sum()
    for extra in [False,True]:
     a=X[:,list(range(1,5))+([col] if extra else [])]
     mu=np.average(a[tr],axis=0,weights=tw);sd=np.sqrt(np.average((a[tr]-mu)**2,axis=0,weights=tw));sd=np.maximum(sd,1e-12)
     z=np.column_stack([np.ones(tr.sum()),(a[tr]-mu)/sd]);zt=np.column_stack([np.ones(te.sum()),(a[te]-mu)/sd])
     reg=np.eye(z.shape[1]);reg[0,0]=0
     b=np.linalg.solve(z.T@(tw[:,None]*z)+reg,z.T@(tw*y[tr]))
     errs[extra].append(float(np.average((y[te]-zt@b)**2,weights=weight[te])))
   result['comparisons'][target+'_'+feature]=dict(slope_per_within_SD=ci([np.mean(slopes[w]) for w in works]),
    mse_base=float(np.mean(errs[False])),mse_extended=float(np.mean(errs[True])),improvement=ci(np.array(errs[False])-np.array(errs[True])),
    per_work=[dict(work=w,mse_base=errs[False][i],mse_extended=errs[True][i],slope=float(np.mean(slopes[w]))) for i,w in enumerate(works)])
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 (OUT/'result.json').write_text(json.dumps(result,indent=2));(OUT/'state.json').write_text('{"status":"COMPLETE"}')
 print(json.dumps({**result,'comparisons':{k:{a:b for a,b in v.items() if a!='per_work'} for k,v in result['comparisons'].items()}},indent=2))
if __name__=='__main__':main()
