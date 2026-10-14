#!/usr/bin/env python3
"""Frozen dimensionless ASAP-to-PianoCoRe P3-R transfer."""

import json, math, sys
from pathlib import Path
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.p3_extension import cosine_cost,bayes_from_cost
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden,continue_path
from p3_r_trigger_audit import centers_and_masses,compare_features,posterior_stats

CONFIG=ROOT/'configs/p3_temporal.yaml'; ACQ=ROOT/'results/p3_pianocore_p3r_acquisition.json'
ASAP=ROOT/'results/p3_r_trigger/result.json'; OUT=ROOT/'results/p3_r_invariant_transfer'
WINDOWS=((10,20),(20,30),(30,40),(40,50)); BETA=8.; RESTART=1e-6; SEED=20260920
FEATURES=('restore_entropy','restore_max_probability','decision_entropy','decision_max_probability',
          'erased_unique_fraction','erased_nonlocal_switch_rate','candidate_max_mass',
          'candidate_mass_entropy','max_log_bf_per_frame','score_margin_per_frame')

def invariant(raw):
    return {'restore_entropy':raw['restore_entropy'],'restore_max_probability':raw['restore_max_probability'],
            'decision_entropy':raw['decision_entropy'],'decision_max_probability':raw['decision_max_probability'],
            'erased_unique_fraction':raw['erased_unique_fraction'],
            'erased_nonlocal_switch_rate':raw['erased_nonlocal_switches']/99.,
            'candidate_max_mass':raw['candidate_max_mass'],'candidate_mass_entropy':raw['candidate_mass_entropy'],
            'max_log_bf_per_frame':raw['max_log_bf']/20.,'score_margin_per_frame':raw['score_margin']/20.}

def vector(features): return np.asarray([features[k] for k in FEATURES],dtype=float)

def extract(cost,posterior,path,transition,score_hop,onset,restore,decision):
    centers,masses=centers_and_masses(posterior,score_hop)
    states,null,scores,endpoints,selected,null_score=compare_features(cost,posterior,centers,transition,score_hop,restore,decision)
    re,rm,rs=posterior_stats(posterior,score_hop); de,dm,ds=posterior_stats(null,score_hop)
    segment=path[onset:restore]; radius=max(len(transition)-1,int(round(.5/score_hop)))
    prob=masses/masses.sum(); ordered=np.sort(scores)
    raw={'restore_entropy':re,'restore_max_probability':rm,'restore_std_seconds':rs,
         'decision_entropy':de,'decision_max_probability':dm,'decision_std_seconds':ds,
         'erased_net_progress_seconds':float((segment[-1]-segment[0])*score_hop),
         'erased_unique_fraction':float(len(np.unique(segment))/len(segment)),
         'erased_nonlocal_switches':int(np.sum(np.abs(np.diff(segment))>radius)),
         'candidate_max_mass':float(masses.max()),
         'candidate_mass_entropy':-float(np.sum(prob*np.log(np.maximum(prob,1e-300))))/math.log(len(prob)),
         'max_log_bf':float(scores[selected]-null_score),'score_margin':float(ordered[-1]-ordered[-2])}
    return invariant(raw),states,selected

def bootstrap(rows,key):
    grouped={}
    for r in rows: grouped.setdefault(r['composer'],[]).append(r[key])
    values=np.asarray([np.mean(v) for v in grouped.values()]); rng=np.random.default_rng(SEED)
    draws=rng.choice(values,(20000,len(values)),replace=True).mean(axis=1)
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(draws,[.025,.975])]}

def main():
    old=json.loads(ASAP.read_text()); train_rows=old['rows']
    x=np.asarray([vector(invariant(r)) for r in train_rows]); y=np.asarray([r['persistent_event'] for r in train_rows],dtype=int)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=SEED))
    model.fit(x,y); fitted=model.predict_proba(x)[:,1]; threshold=float(np.quantile(fitted[y==0],.95,method='higher'))
    cfg=yaml.safe_load(CONFIG.read_text()); hop=float(cfg['hop_sec']); transition=np.asarray(cfg['bayes']['transition'])
    acquisition=json.loads(ACQ.read_text()); rows=[]
    with threadpool_limits(limits=1):
      for number,case in enumerate(acquisition['rows'],1):
        pn=load_notes(ROOT/case['performance_path']); sn=load_notes(ROOT/case['score_path'])
        pd,sd=max(n[1] for n in pn),max(n[1] for n in sn); score_hop=hop*sd/pd
        base=cosine_cost(notes_to_chroma_stream(pn,60.,hop),notes_to_chroma_stream(sn,sd,score_hop))
        intact=bayes_from_cost(base,transition,BETA,RESTART); end=len(base)
        for start_sec,restore_sec in WINDOWS:
          onset,restore=int(start_sec/hop),int(restore_sec/hop); decision,eval_end=restore+20,restore+100
          erased=base.copy(); erased[onset:restore]=erased[onset:restore].mean(axis=1,keepdims=True)
          natural=bayes_from_cost(erased,transition,BETA,RESTART); nb=burden(natural,intact,score_hop,decision,eval_end)
          posterior,prefix=posterior_until(erased,restore,transition,BETA,RESTART)
          if not np.array_equal(prefix,natural[:restore]): raise AssertionError('prefix mismatch')
          features,states,selected=extract(erased,posterior,prefix,transition,score_hop,onset,restore,decision)
          probability=float(model.predict_proba(vector(features)[None,:])[0,1]); trigger=probability>=threshold
          burdens=[]
          for state in states:
            path=natural.copy(); path[decision:]=continue_path(erased,state.copy(),transition,decision,end)
            burdens.append(burden(path,intact,score_hop,decision,eval_end))
          sb=burdens[selected] if trigger else nb; ob=min([nb]+burdens)
          iposterior,iprefix=posterior_until(base,restore,transition,BETA,RESTART)
          if not np.array_equal(iprefix,intact[:restore]): raise AssertionError('intact prefix mismatch')
          ifeatures,istates,iselected=extract(base,iposterior,iprefix,transition,score_hop,onset,restore,decision)
          ip=float(model.predict_proba(vector(ifeatures)[None,:])[0,1]); it=ip>=threshold
          if it:
            sp=intact.copy(); sp[decision:]=continue_path(base,istates[iselected].copy(),transition,decision,end)
            ib=burden(sp,intact,score_hop,decision,eval_end)
          else: ib=0.
          rows.append({'composer':case['composer'],'id':case['id'],'window_sec':[start_sec,restore_sec],
                       'persistent_event':nb>.5,'probability':probability,'trigger':trigger,
                       'natural_burden':nb,'selector_burden':sb,'oracle_burden':ob,
                       'selector_improvement':nb-sb,'oracle_improvement':nb-ob,
                       'safety_probability':ip,'safety_trigger':it,'safety_burden':ib})
        print(f"[{number}/{len(acquisition['rows'])}] {case['id']} {case['composer']}")
    labels=np.asarray([r['persistent_event'] for r in rows]); scores=np.asarray([r['probability'] for r in rows]); pred=np.asarray([r['trigger'] for r in rows])
    prevalence=float(labels.mean()); auc=float(roc_auc_score(labels,scores)) if labels.any() and (~labels).any() else None
    ap=float(average_precision_score(labels,scores)) if labels.any() else None
    recall=float(np.sum(pred&labels)/np.sum(labels)) if labels.any() else 0.; fpr=float(np.sum(pred&~labels)/np.sum(~labels)) if (~labels).any() else 0.
    affected=len(set(r['composer'] for r in rows if r['persistent_event']))
    names=('natural_burden','selector_burden','oracle_burden','selector_improvement','oracle_improvement','safety_burden')
    metrics={k:bootstrap(rows,k) for k in names}; nm=metrics['natural_burden']['mean']; oi=metrics['oracle_improvement']; si=metrics['selector_improvement']
    gates={'coverage':len(rows)==120 and len(set(r['composer'] for r in rows))==30,
           'incidence':int(labels.sum())>=15 and affected>=10,
           'trigger':auc is not None and auc>=.70 and ap>=2*prevalence and recall>=.35 and fpr<=.10,
           'oracle':oi['mean']>=.2*nm and oi['ci95'][0]>0,
           'selector':si['mean']>=.2*nm and si['ci95'][0]>0,
           'fraction_of_oracle':oi['mean']>0 and si['mean']>=.5*oi['mean'],
           'safety':metrics['safety_burden']['mean']<=.1 and np.mean([r['safety_burden']>.5 for r in rows])<=.05}
    gates['all']=bool(all(gates.values()))
    result={'scope':'Frozen ASAP dimensionless-trigger transfer to PianoCoRe development; not natural P3',
            'features':FEATURES,'asap_fitted_threshold':threshold,'coverage':{'composers':30,'interventions':len(rows)},
            'incidence':{'events':int(labels.sum()),'affected_composers':affected,'prevalence':prevalence},
            'trigger':{'auroc':auc,'average_precision':ap,'recall':recall,'false_positive_rate':fpr,'positive':int(pred.sum())},
            'metrics':metrics,'safety_violation_rate':float(np.mean([r['safety_burden']>.5 for r in rows])),
            'gates':gates,'decision':'download_preserved_validation' if gates['all'] else 'stop_before_preserved_validation','rows':rows}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__': main()
