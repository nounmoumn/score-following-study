"""Descriptive terminal-boundary audit; does not replace frozen inference."""
import json,sys
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import read_csv,load_case
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior
out=ROOT/'results/p3_mechanism_20260918_003738'
cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
c=next(c for c in read_csv(out/'manifest.csv') if c['composition']=='Chopin/Ballades_4')
u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);p=np.zeros(len(v));p[0]=1;b=cfg['bayes'];records=[]
for i,row in enumerate(cost[:400]):
    if i:p=predict_prior(p,b['transition'],b['restart_probability'])
    if i<300:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
    if i in [298,299,300,309,319,349,399]:
        j=int(round(truth[i]));records.append(dict(frame=i,map_frame=int(p.argmax()),terminal_frame=len(v)-1,terminal_mass=float(p[-1]),max_mass=float(p.max()),truth_frame=float(truth[i]),correct_region_mass=float(p[max(0,j-5):j+6].sum())))
result=dict(composition=c['composition'],hop_score_sec=hv,frames=records)
(out/'boundary_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
