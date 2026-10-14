"""Predeclared equal-work diagnostics and leave-work-out incremental explanation."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/online_ambiguity_diagnosis_v2'
CONTRASTS={
 'representation_bayes':{'bayes_pitch':1,'bayes_chroma':-1},
 'representation_dtw':{'dtw_pitch':1,'dtw_chroma':-1},
 'structure_pitch':{'bayes_pitch':1,'dtw_pitch':-1},
 'structure_chroma':{'bayes_chroma':1,'dtw_chroma':-1},
 'context_chroma':{'lag_chroma':1,'bayes_chroma':-1},
 'context_pitch':{'lag_pitch':1,'bayes_pitch':-1},
 'context_representation_interaction':{'lag_pitch':1,'lag_chroma':-1,'bayes_pitch':-1,'bayes_chroma':1}}

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 if (OUT/'result.json').exists():raise SystemExit('Refuse overwrite')
 frozen=json.loads((OUT/'freeze.json').read_text())
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 cases=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
 allcases=cases
 missing=[dict(case=c['case'],time=r['time'],truth=r['truth']) for c in cases for r in c['rows'] if r['A'] is None]
 assert len(missing)==3
 cases=[{**c,'rows':[r for r in c['rows'] if r['A'] is not None]} for c in cases]
 works=sorted({c['work'] for c in cases});W=len(works)
 wi={w:i for i,w in enumerate(works)}
 counts={w:sum(c['work']==w for c in cases) for w in works}
 allrows=[r for c in cases for r in c['rows']]
 X=np.array([[r['A']]+r['controls'] for r in allrows])
 weights=np.concatenate([np.full(len(c['rows']),1/(W*counts[c['work']]*len(c['rows']))) for c in cases])
 workidx=np.concatenate([np.full(len(c['rows']),wi[c['work']]) for c in cases])
 cx=X.copy();offset=0
 for c in cases:
  sl=slice(offset,offset+len(c['rows']));cx[sl]-=cx[sl].mean(0);offset+=len(c['rows'])
 scale=np.sqrt(np.sum(weights[:,None]*cx**2,axis=0));scale=np.maximum(scale,1e-12)
 z=cx/scale
 rng=np.random.default_rng(20260923);sample=rng.integers(0,W,(5000,W))
 def ci(v):
  v=np.array(v);return dict(mean=float(v.mean()),bootstrap95=np.quantile(v[sample].mean(1),[.025,.975]).tolist(),positive_works=int((v>0).sum()),negative_works=int((v<0).sum()))
 def workmeans(v):return [float(np.average(v[workidx==i],weights=weights[workidx==i])) for i in range(W)]
 result={'scope':'opened development, causal MIDI replay, no independent confirmation','cases':len(cases),'works':W,'rows':len(allrows),'missing_A':missing,'models':{},'contrasts':{}}
 for model in allrows[0]['correct']:
  result['models'][model]={'accuracy':ci(workmeans(np.array([r['correct'][model] for r in allrows]))),
                          'absolute_error_score_sec':ci(workmeans(np.array([r['errors'][model] for r in allrows])))}
  result['models'][model]['accuracy_all_4889']=ci([np.mean([np.mean([r['correct'][model] for r in c['rows']]) for c in allcases if c['work']==w]) for w in works])
 for name,coeff in CONTRASTS.items():
  y=np.array([sum(v*r['correct'][k] for k,v in coeff.items()) for r in allrows],dtype=float)
  cy=y.copy();offset=0;slopes={w:[] for w in works}
  for c in cases:
   sl=slice(offset,offset+len(c['rows']));a=cx[sl,0];sd=a.std()
   slopes[c['work']].append(float(np.mean(a/sd*y[sl])) if sd>0 else 0.)
   cy[sl]-=cy[sl].mean();offset+=len(c['rows'])
  xx=np.array([z[workidx==i].T@(weights[workidx==i,None]*z[workidx==i]) for i in range(W)])
  xy=np.array([z[workidx==i].T@(weights[workidx==i]*cy[workidx==i]) for i in range(W)])
  b=np.linalg.pinv(xx.sum(0))@xy.sum(0)
  boot=np.array([(np.linalg.pinv(xx[s].sum(0))@xy[s].sum(0))[0] for s in sample])
  mse={False:[],True:[]}
  for i in range(W):
   train=workidx!=i;test=~train
   for useA in [False,True]:
    a=X if useA else X[:,1:]
    tw=weights[train];tw=tw/tw.sum()*train.sum()
    mean=np.average(a[train],axis=0,weights=tw)
    std=np.sqrt(np.average((a[train]-mean)**2,axis=0,weights=tw));std=np.maximum(std,1e-12)
    tr=np.column_stack([np.ones(train.sum()),(a[train]-mean)/std])
    te=np.column_stack([np.ones(test.sum()),(a[test]-mean)/std])
    penalty=np.eye(tr.shape[1]);penalty[0,0]=0
    beta=np.linalg.solve(tr.T@(tw[:,None]*tr)+penalty,tr.T@(tw*y[train]))
    mse[useA].append(float(np.average((y[test]-te@beta)**2,weights=weights[test])))
  result['contrasts'][name]={'gain':ci(workmeans(y)),
       'within_case_A_slope_per_SD':ci([np.mean(slopes[w]) for w in works]),
       'controlled_A_coefficient_per_within_SD':{'value':float(b[0]),'bootstrap95':np.quantile(boot,[.025,.975]).tolist()},
       'LOOW_MSE_controls':float(np.mean(mse[False])), 'LOOW_MSE_controls_A':float(np.mean(mse[True])),
       'LOOW_MSE_improvement_from_A':ci(np.array(mse[False])-np.array(mse[True])),
       'per_work':[dict(work=w,gain=workmeans(y)[i],slope=float(np.mean(slopes[w])),mse_controls=mse[False][i],mse_A=mse[True][i]) for i,w in enumerate(works)]}
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 write(OUT/'result.json',result)
 write(OUT/'state.json',{'status':'COMPLETE'})
 print(json.dumps({**result,'contrasts':{k:{a:b for a,b in v.items() if a!='per_work'} for k,v in result['contrasts'].items()}},indent=2))
if __name__=='__main__':main()
