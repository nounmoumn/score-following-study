import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/tracker_path_diagnosis'
def main():
 if (OUT/'result.json').exists():raise SystemExit('Refuse overwrite')
 h=json.loads((OUT/'freeze.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
 works=sorted({c['work'] for c in cases});W=len(works);rows=[r for c in cases for r in c['rows']]
 ids=np.concatenate([np.full(len(c['rows']),works.index(c['work'])) for c in cases])
 comp=np.concatenate([np.full(len(c['rows']),c['composer']) for c in cases])
 counts={w:sum(c['work']==w for c in cases) for w in works}
 weight=np.concatenate([np.full(len(c['rows']),1/(W*counts[c['work']]*len(c['rows']))) for c in cases])
 B=np.array([r['candidate']['base'] for r in rows]);S=np.array([r['candidate']['score'] for r in rows])
 sham=np.concatenate([np.roll(np.array([r['candidate']['score'] for r in c['rows']]),len(c['rows'])//3,axis=0) for c in cases])
 designs={'base':B,'real':np.column_stack([B,S]),'sham':np.column_stack([B,sham])}
 sample=np.random.default_rng(20260923).integers(0,W,(5000,W))
 def ci(v):
  v=np.asarray(v);return dict(mean=float(v.mean()),bootstrap95=np.quantile(v[sample].mean(1),[.025,.975]).tolist(),positive_works=int((v>0).sum()),negative_works=int((v<0).sum()))
 result={'scope':'adaptive opened-data, representative maximum joint probability paths under same kernel, not independent confirmation','rows':len(rows),'works':W,'comparisons':{},'coverage':{}}
 for target in ['representation','context']:
  y=np.array([r['correct']['bayes_pitch']-r['correct']['bayes_chroma'] if target=='representation' else r['correct']['lag_chroma']-r['correct']['bayes_chroma'] for r in rows])
  for grouping,groups in [('LOOW',ids),('LOCO',comp)]:
   preds={k:np.zeros(len(rows)) for k in designs}
   for g in np.unique(groups):
    te=groups==g;tr=~te;tw=weight[tr]/weight[tr].sum()*tr.sum()
    for name,a in designs.items():
     mu=np.average(a[tr],axis=0,weights=tw);sd=np.maximum(np.sqrt(np.average((a[tr]-mu)**2,axis=0,weights=tw)),1e-12)
     z=np.column_stack([np.ones(tr.sum()),(a[tr]-mu)/sd]);zt=np.column_stack([np.ones(te.sum()),(a[te]-mu)/sd])
     reg=np.eye(z.shape[1]);reg[0,0]=0
     b=np.linalg.solve(z.T@(tw[:,None]*z)+reg,z.T@(tw*y[tr]));preds[name][te]=zt@b
   mse={k:np.array([np.average((y[ids==i]-v[ids==i])**2,weights=weight[ids==i]) for i in range(W)]) for k,v in preds.items()}
   result['comparisons'][target+'_'+grouping]={'mse':{k:float(v.mean()) for k,v in mse.items()},'real_over_base':ci(mse['base']-mse['real']),'real_over_sham':ci(mse['sham']-mse['real']),
       'per_work':[dict(work=w,**{k:float(v[i]) for k,v in mse.items()}) for i,w in enumerate(works)]}
 wrong=[r for r in rows if not r['correct']['bayes_chroma']]
 covered=[r for r in wrong if abs((r['candidate']['rival']+1)*.1-r['truth'])<=.5+1e-12]
 result['coverage']={'wrong':len(wrong),'rival_near_truth':len(covered),'covered_pitch_fixed':sum(r['correct']['bayes_pitch'] for r in covered)}
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 (OUT/'result.json').write_text(json.dumps(result,indent=2));(OUT/'state.json').write_text('{"status":"COMPLETE"}')
 print(json.dumps({**result,'comparisons':{k:{a:b for a,b in v.items() if a!='per_work'} for k,v in result['comparisons'].items()}},indent=2))
if __name__=='__main__':main()
