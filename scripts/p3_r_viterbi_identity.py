#!/usr/bin/env python3
"""Four-second coherent Viterbi candidate identity audit."""

import json,sys
from pathlib import Path
import numpy as np,yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.p3_extension import cosine_cost,bayes_from_cost
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden,continue_path
from p3_r_trigger_audit import centers_and_masses,compare_features
from p3_r_invariant_transfer import extract,vector,bootstrap
from p3_r_branch_policy import fit_fault

ACQ=ROOT/'results/p3_pianocore_p3r_acquisition.json'; CONFIG=ROOT/'configs/p3_temporal.yaml'; OUT=ROOT/'results/p3_r_viterbi_identity'
WINDOWS=((10,20),(20,30),(30,40),(40,50)); BETA=8.; RESTART=1e-6

def viterbi_scores(cost,centers,transition,score_hop,restore,decision):
    n=cost.shape[1]; radius=max(1,int(round(2./score_hop))); neglog=-np.log(np.maximum(transition,1e-300)); scores=[]
    for center in centers:
        dp=np.full(n,np.inf); dp[max(0,center-radius):min(n,center+radius+1)]=0.
        for frame in range(restore,decision):
            nxt=np.full(n,np.inf)
            for step,penalty in enumerate(neglog):
                if step==0: candidate=dp+penalty
                else:
                    candidate=np.full(n,np.inf); candidate[step:]=dp[:-step]+penalty
                    candidate[-1]=min(candidate[-1],float(np.min(dp[-step:]+penalty)))
                nxt=np.minimum(nxt,candidate)
            dp=nxt+BETA*(cost[frame]-cost[frame].min())
        scores.append(float(np.min(dp))/(decision-restore))
    return np.asarray(scores)

def main():
    cfg=yaml.safe_load(CONFIG.read_text()); hop=float(cfg['hop_sec']); transition=np.asarray(cfg['bayes']['transition']); fault_model,fault_threshold=fit_fault()
    cases=json.loads(ACQ.read_text())['rows']; rows=[]
    with threadpool_limits(limits=1):
      for number,case in enumerate(cases,1):
        pn=load_notes(ROOT/case['performance_path']); sn=load_notes(ROOT/case['score_path']); pd,sd=max(n[1] for n in pn),max(n[1] for n in sn); sh=hop*sd/pd
        base=cosine_cost(notes_to_chroma_stream(pn,60.,hop),notes_to_chroma_stream(sn,sd,sh)); intact=bayes_from_cost(base,transition,BETA,RESTART); end=len(base)
        for start,stop in WINDOWS:
          onset,restore=int(start/hop),int(stop/hop); trigger_time,decision,eval_end=restore+20,restore+40,restore+100
          erased=base.copy(); erased[onset:restore]=erased[onset:restore].mean(axis=1,keepdims=True); natural=bayes_from_cost(erased,transition,BETA,RESTART)
          event_burden=burden(natural,intact,sh,trigger_time,eval_end); nb=burden(natural,intact,sh,decision,eval_end)
          post,prefix=posterior_until(erased,restore,transition,BETA,RESTART); inv,_,_=extract(erased,post,prefix,transition,sh,onset,restore,trigger_time)
          fault_probability=float(fault_model.predict_proba(vector(inv)[None,:])[0,1]); trigger=fault_probability>=fault_threshold
          centers,_=centers_and_masses(post,sh); states,_,_,_,_,_=compare_features(erased,post,centers,transition,sh,restore,decision)
          vs=viterbi_scores(erased,centers,transition,sh,restore,decision); selected=int(np.argmin(vs)); burdens=[]
          for state in states:
            path=natural.copy(); path[decision:]=continue_path(erased,state.copy(),transition,decision,end); burdens.append(burden(path,intact,sh,decision,eval_end))
          sb=burdens[selected] if trigger else nb; ob=min([nb]+burdens)
          ipost,iprefix=posterior_until(base,restore,transition,BETA,RESTART); iinv,_,_=extract(base,ipost,iprefix,transition,sh,onset,restore,trigger_time)
          safety_trigger=float(fault_model.predict_proba(vector(iinv)[None,:])[0,1])>=fault_threshold
          icenters,_=centers_and_masses(ipost,sh); istates,_,_,_,_,_=compare_features(base,ipost,icenters,transition,sh,restore,decision)
          ivs=viterbi_scores(base,icenters,transition,sh,restore,decision); iselected=int(np.argmin(ivs))
          if safety_trigger:
            path=intact.copy(); path[decision:]=continue_path(base,istates[iselected].copy(),transition,decision,end); ib=burden(path,intact,sh,decision,eval_end)
          else: ib=0.
          rows.append({'composer':case['composer'],'id':case['id'],'window_sec':[start,stop],'persistent_event':event_burden>.5,
                       'trigger':trigger,'natural_burden':nb,'selector_burden':sb,'oracle_burden':ob,
                       'selector_improvement':nb-sb,'oracle_improvement':nb-ob,'safety_burden':ib,'safety_trigger':safety_trigger})
        print(f"[{number}/{len(cases)}] {case['id']}")
    metrics={k:bootstrap(rows,k) for k in ('natural_burden','selector_burden','oracle_burden','selector_improvement','oracle_improvement','safety_burden')}
    nm=metrics['natural_burden']['mean']; si=metrics['selector_improvement']; oi=metrics['oracle_improvement']; acted=[r for r in rows if r['trigger']]
    gates={'recovery':si['mean']>=.2*nm and si['ci95'][0]>0,'fraction_of_oracle':oi['mean']>0 and si['mean']>=.5*oi['mean'],
           'action_direction':sum(r['selector_improvement']>0 for r in acted)>sum(r['selector_improvement']<0 for r in acted),
           'safety':metrics['safety_burden']['mean']<=.1 and np.mean([r['safety_burden']>.5 for r in rows])<=.05}; gates['all']=bool(all(gates.values()))
    result={'scope':'Opened PianoCoRe development long-context Viterbi identity; preserved validation sealed','decision_delay_sec':4.,
            'events':sum(r['persistent_event'] for r in rows),'affected_composers':len(set(r['composer'] for r in rows if r['persistent_event'])),
            'actions':len(acted),'helpful_actions':sum(r['selector_improvement']>0 for r in acted),'harmful_actions':sum(r['selector_improvement']<0 for r in acted),
            'safety_actions':sum(r['safety_trigger'] for r in rows),'metrics':metrics,'safety_violation_rate':float(np.mean([r['safety_burden']>.5 for r in rows])),
            'gates':gates,'decision':'viterbi_identity_feasible_but_validation_still_blocked' if gates['all'] else 'stop_candidate_identity_family','rows':rows}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

if __name__=='__main__': main()
