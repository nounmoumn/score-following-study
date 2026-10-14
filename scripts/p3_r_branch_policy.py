#!/usr/bin/env python3
"""Composer-held-out null-aware branch action policy on opened PianoCoRe dev."""

import json,math,sys
from pathlib import Path
import numpy as np,yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.p3_extension import cosine_cost,bayes_from_cost
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden,continue_path
from p3_r_trigger_audit import centers_and_masses,compare_features
from p3_r_invariant_transfer import invariant,vector,bootstrap

ACQ=ROOT/'results/p3_pianocore_p3r_acquisition.json'; ASAP=ROOT/'results/p3_r_trigger/result.json'
CONFIG=ROOT/'configs/p3_temporal.yaml'; OUT=ROOT/'results/p3_r_branch_policy'; WINDOWS=((10,20),(20,30),(30,40),(40,50))
FEATURES=('evidence_advantage','evidence_rank','initial_mass','mass_rank','endpoint_entropy',
          'endpoint_max_probability','center_null_fraction','endpoint_null_fraction')

def stats(p):
    q=p[p>0]; return -float(np.sum(q*np.log(q)))/math.log(len(p)),float(p.max())

def branch_online(cost,posterior,transition,score_hop,restore,decision):
    centers,masses=centers_and_masses(posterior,score_hop)
    states,null,scores,endpoints,_,null_score=compare_features(cost,posterior,centers,transition,score_hop,restore,decision)
    n=len(posterior); erank=np.argsort(np.argsort(scores))/(len(scores)-1); mrank=np.argsort(np.argsort(masses))/(len(masses)-1)
    null_restore=int(np.argmax(posterior)); null_end=int(np.argmax(null)); records=[]
    for j,state in enumerate(states):
        ent,mx=stats(state)
        records.append({'evidence_advantage':float((scores[j]-null_score)/20.),'evidence_rank':float(erank[j]),
          'initial_mass':float(masses[j]),'mass_rank':float(mrank[j]),'endpoint_entropy':ent,
          'endpoint_max_probability':mx,'center_null_fraction':abs(centers[j]-null_restore)/n,
          'endpoint_null_fraction':abs(endpoints[j]-null_end)/n})
    return states,records

def fit_fault():
    rows=json.loads(ASAP.read_text())['rows']; x=np.asarray([vector(invariant(r)) for r in rows]); y=np.asarray([r['persistent_event'] for r in rows])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=20260920)); model.fit(x,y)
    score=model.predict_proba(x)[:,1]; threshold=float(np.quantile(score[~y],.95,method='higher')); return model,threshold

def main():
    cfg=yaml.safe_load(CONFIG.read_text()); hop=float(cfg['hop_sec']); transition=np.asarray(cfg['bayes']['transition'])
    fault_model,fault_threshold=fit_fault(); cases=json.loads(ACQ.read_text())['rows']; events=[]; branches=[]
    with threadpool_limits(limits=1):
      for number,case in enumerate(cases,1):
        pn=load_notes(ROOT/case['performance_path']); sn=load_notes(ROOT/case['score_path']); pd,sd=max(n[1] for n in pn),max(n[1] for n in sn); sh=hop*sd/pd
        base=cosine_cost(notes_to_chroma_stream(pn,60.,hop),notes_to_chroma_stream(sn,sd,sh)); intact=bayes_from_cost(base,transition,8.,1e-6); end=len(base)
        for start,stop in WINDOWS:
          onset,restore=int(start/hop),int(stop/hop); decision,eval_end=restore+20,restore+100
          erased=base.copy(); erased[onset:restore]=erased[onset:restore].mean(axis=1,keepdims=True); natural=bayes_from_cost(erased,transition,8.,1e-6)
          nb=burden(natural,intact,sh,decision,eval_end); post,prefix=posterior_until(erased,restore,transition,8.,1e-6)
          from p3_r_invariant_transfer import extract
          inv,_,_=extract(erased,post,prefix,transition,sh,onset,restore,decision); fault_prob=float(fault_model.predict_proba(vector(inv)[None,:])[0,1])
          states,online=branch_online(erased,post,transition,sh,restore,decision); improvements=[]
          for j,(state,feat) in enumerate(zip(states,online)):
            path=natural.copy(); path[decision:]=continue_path(erased,state.copy(),transition,decision,end); bb=burden(path,intact,sh,decision,eval_end); imp=nb-bb; improvements.append(imp)
            branches.append({'composer':case['composer'],'event_key':f"{case['id']}:{start}",'kind':'erased','branch':j,
                             'beneficial':imp>.5,'improvement':imp,**feat})
          ipost,iprefix=posterior_until(base,restore,transition,8.,1e-6); iinv,_,_=extract(base,ipost,iprefix,transition,sh,onset,restore,decision)
          safety_fault_prob=float(fault_model.predict_proba(vector(iinv)[None,:])[0,1]); istates,ionline=branch_online(base,ipost,transition,sh,restore,decision)
          safety_burdens=[]
          for j,(state,feat) in enumerate(zip(istates,ionline)):
            path=intact.copy(); path[decision:]=continue_path(base,state.copy(),transition,decision,end); ib=burden(path,intact,sh,decision,eval_end); safety_burdens.append(ib)
            branches.append({'composer':case['composer'],'event_key':f"{case['id']}:{start}:intact",'kind':'intact','branch':j,
                             'beneficial':False,'improvement':-ib,**feat})
          events.append({'composer':case['composer'],'id':case['id'],'window_sec':[start,stop],'natural_burden':nb,
                         'oracle_improvement':max([0.]+improvements),'fault_trigger':fault_prob>=fault_threshold,
                         'safety_fault_trigger':safety_fault_prob>=fault_threshold,'branch_improvements':improvements,
                         'safety_burdens':safety_burdens})
        print(f"[{number}/{len(cases)}] {case['id']}")
    x=np.asarray([[b[k] for k in FEATURES] for b in branches]); y=np.asarray([b['beneficial'] for b in branches]); groups=np.asarray([b['composer'] for b in branches])
    probability=np.zeros(len(branches)); thresholds={}
    for composer in sorted(set(groups)):
      test=groups==composer; train=~test; model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=20260920))
      model.fit(x[train],y[train]); train_score=model.predict_proba(x[train])[:,1]; thresholds[composer]=float(np.quantile(train_score[~y[train]],.99,method='higher')); probability[test]=model.predict_proba(x[test])[:,1]
    lookup={(b['event_key'],b['branch']):probability[i] for i,b in enumerate(branches)}; actions=[]
    for event in events:
      key=f"{event['id']}:{event['window_sec'][0]}"; probs=np.asarray([lookup[(key,j)] for j in range(10)]); j=int(np.argmax(probs)); threshold=thresholds[event['composer']]
      act=event['fault_trigger'] and probs[j]>=threshold; improvement=event['branch_improvements'][j] if act else 0.
      ikey=key+':intact'; iprobs=np.asarray([lookup[(ikey,k)] for k in range(10)]); ij=int(np.argmax(iprobs)); iact=event['safety_fault_trigger'] and iprobs[ij]>=threshold; ib=event['safety_burdens'][ij] if iact else 0.
      actions.append({'composer':event['composer'],'id':event['id'],'window_sec':event['window_sec'],'natural_burden':event['natural_burden'],
                      'oracle_improvement':event['oracle_improvement'],'selector_improvement':improvement,'selector_burden':event['natural_burden']-improvement,
                      'action':act,'chosen_probability':float(probs[j]),'fold_threshold':threshold,'safety_action':iact,'safety_burden':ib})
    metrics={k:bootstrap(actions,k) for k in ('natural_burden','oracle_improvement','selector_improvement','selector_burden','safety_burden')}
    nm=metrics['natural_burden']['mean']; oi=metrics['oracle_improvement']; si=metrics['selector_improvement']; acted=[a for a in actions if a['action']]
    gates={'recovery':si['mean']>=.2*nm and si['ci95'][0]>0,'fraction_of_oracle':oi['mean']>0 and si['mean']>=.5*oi['mean'],
           'action_direction':sum(a['selector_improvement']>0 for a in acted)>sum(a['selector_improvement']<0 for a in acted),
           'safety':metrics['safety_burden']['mean']<=.1 and np.mean([a['safety_burden']>.5 for a in actions])<=.05}
    gates['all']=bool(all(gates.values()))
    result={'scope':'PianoCoRe opened-development branch-policy LOCO feasibility; preserved validation remains sealed','features':FEATURES,
            'branch_records':len(branches),'beneficial_branches':int(y.sum()),'actions':len(acted),'helpful_actions':sum(a['selector_improvement']>0 for a in acted),
            'harmful_actions':sum(a['selector_improvement']<0 for a in acted),'safety_actions':sum(a['safety_action'] for a in actions),
            'metrics':metrics,'safety_violation_rate':float(np.mean([a['safety_burden']>.5 for a in actions])),'gates':gates,
            'decision':'feasible_but_upstream_incidence_still_blocks_validation' if gates['all'] else 'branch_policy_not_supported','rows':actions}
    encode=lambda value: value.item() if isinstance(value,np.generic) else str(value)
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'result.json').write_text(json.dumps(result,indent=2,default=encode),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,default=encode))

if __name__=='__main__': main()
