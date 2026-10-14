"""Independent saved-output arithmetic audit for ambiguity_stress_test_utility."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ambiguity_stress_test_utility'
def pred(x,fit):
 x=np.asarray(x,float);mu=np.asarray(fit['mean']);sd=np.asarray(fit['sd']);b=np.asarray(fit['coefficients'])
 return b[0]+((x-mu)/sd)@b[1:]
def case_work_mean(values,cases,work):
 return float(np.mean([np.mean(values[c['case']]) for c in cases if c['work']==work]))
def main():
 result=json.loads((OUT/'result.json').read_text());cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
 works=sorted({c['work'] for c in cases});yb={};pb={};pa={};lc={};lp={}
 for c in cases:
  state=np.array([r['state'] for r in c['rows']]);a=np.array([r['A'] for r in c['rows']]);lc[c['case']]=np.array([r['chroma'] for r in c['rows']]);lp[c['case']]=np.array([r['pitch'] for r in c['rows']]);yb[c['case']]=lc[c['case']]-lp[c['case']]
  pb[c['case']]=pred(state,result['fits']['state']);pa[c['case']]=pred(np.column_stack([state,a]),result['fits']['state_A'])
 mb=[];ma=[];actual=[];pmb=[];pma=[];cl=[];pl=[]
 for w in works:
  mb.append(case_work_mean({k:(yb[k]-pb[k])**2 for k in yb},cases,w));ma.append(case_work_mean({k:(yb[k]-pa[k])**2 for k in yb},cases,w))
  actual.append(case_work_mean(yb,cases,w));pmb.append(case_work_mean(pb,cases,w));pma.append(case_work_mean(pa,cases,w));cl.append(case_work_mean(lc,cases,w));pl.append(case_work_mean(lp,cases,w))
 mb=np.array(mb);ma=np.array(ma);actual=np.array(actual);pmb=np.array(pmb);pma=np.array(pma);cl=np.array(cl);pl=np.array(pl);oracle=np.minimum(cl,pl)
 saved=result['primary_mse'];assert abs(mb.mean()-saved['state'])<1e-12 and abs(ma.mean()-saved['state_A'])<1e-12
 assert np.allclose(mb-ma,[x['improvement'] for x in saved['per_work']],atol=1e-12,rtol=0)
 rg=np.where(result['selection_regret']['global_choice']=='pitch',pl,cl)-oracle;rs=np.where(pmb>=0,pl,cl)-oracle;ra=np.where(pma>=0,pl,cl)-oracle
 assert abs(rg.mean()-result['selection_regret']['global']['mean'])<1e-12
 assert abs(rs.mean()-result['selection_regret']['state']['mean'])<1e-12 and abs(ra.mean()-result['selection_regret']['state_A']['mean'])<1e-12
 freeze=json.loads((OUT/'freeze.json').read_text());drift=[p for p,h in freeze.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not drift
 audit={'status':'PASS','cases':len(cases),'works':len(works),'rows':sum(len(c['rows']) for c in cases),'max_mse_residual':float(max(abs(mb.mean()-saved['state']),abs(ma.mean()-saved['state_A']))),'state_and_state_A_same_work_choices':bool(np.array_equal(pmb>=0,pma>=0)),'state_A_minus_global_mean_regret':float(ra.mean()-rg.mean()),'hash_drift':drift}
 (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
