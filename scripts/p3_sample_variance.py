"""Planning-only variance scenarios from existing development fold losses."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import nct,t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from p3_temporal import read_csv,write_csv

def power(n,d):return float(nct.sf(t.ppf(.95,n-1),n-1,d*np.sqrt(n)))

def main():
    out=ROOT/'results/p3_onset_development/sample_variance';out.mkdir(exist_ok=True)
    sources={'current_B':ROOT/'results/p3_onset_development/current_position/work_losses.csv',
             'multi_M':ROOT/'results/p3_onset_development/multicandidate/work_losses.csv'}
    rows=[]
    for name,path in sources.items():
        data=read_csv(path);column='B' if name=='current_B' else 'M';baseline='C' if name=='current_B' else 'constant'
        values=np.array([float(r[baseline+'_brier'])-float(r[column+'_brier']) for r in data])
        rows.append(dict(model=name,works=len(values),observed_mean=float(values.mean()),sd=float(values.std(ddof=1)),
                         median=float(np.median(values)),positive=int((values>0).sum())))
    mde=.004 # approximately 5% of fixed constant-model development Brier 0.0787; illustrative, not selected from a positive effect.
    conservative_sigma=max(r['sd'] for r in rows)
    sensitivity=[]
    for sigma in [r['sd'] for r in rows]+[conservative_sigma*1.5,conservative_sigma*2]:
        d=mde/sigma
        required=next((n for n in range(3,10001) if power(n,d)>=.8),None)
        sensitivity.append(dict(raw_improvement=mde,sigma=sigma,standardized_d=d,works_for80power=required,power_with25=power(25,d)))
    summary=dict(raw_improvement_scenario=mde,source_summary=rows,sensitivity=sensitivity,
        caveat='Development fold-loss dispersion is exploratory and possibly optimistic; effect .004 is illustrative and is not evidence of actual benefit. Paired-t normal independent-work approximation; composer dependence and rare-event power not modeled.')
    write_csv(out/'variance_sources.csv',rows);write_csv(out/'sensitivity.csv',sensitivity)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
