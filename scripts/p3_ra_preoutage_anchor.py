#!/usr/bin/env python3
"""Known-outage P3-Ra recovery using only the pre-outage temporal anchor."""

import csv,json,sys
from pathlib import Path
import numpy as np,yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.p3_extension import cosine_cost,bayes_from_cost
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden,continue_path
from p3_r_trigger_audit import centers_and_masses,compare_features

ASAP_CFG=ROOT/'configs/p3_asap_intervention.yaml'; TEMPORAL=ROOT/'configs/p3_temporal.yaml'
ACQ=ROOT/'results/p3_pianocore_p3r_acquisition.json'; OUT=ROOT/'results/p3_ra_preoutage_anchor'
WINDOWS=((10,20),(20,30),(30,40),(40,50)); BETA=8.; RESTART=1e-6; SEED=20260920

def cluster_bootstrap(rows,key):
    grouped={}
    for r in rows: grouped.setdefault(r['group'],[]).append(r[key])
    values=np.asarray([np.mean(v) for v in grouped.values()]); rng=np.random.default_rng(SEED)
    draws=rng.choice(values,(20000,len(values)),replace=True).mean(axis=1)
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(draws,[.025,.975])]}

def outage(cost,onset,restore): return bool(np.all(np.std(cost[onset:restore],axis=1)<=1e-12))

def evaluate(base,score_hop,transition,corpus,group,identifier):
    intact=bayes_from_cost(base,transition,BETA,RESTART); end=len(base); rows=[]
    for start,stop in WINDOWS:
      onset,restore=int(start*10),int(stop*10); decision,eval_end=restore+20,restore+100
      erased=base.copy(); erased[onset:restore]=erased[onset:restore].mean(axis=1,keepdims=True)
      if not outage(erased,onset,restore): raise AssertionError('erasure outage not detected')
      intact_false=outage(base,onset,restore)
      natural=bayes_from_cost(erased,transition,BETA,RESTART); nb=burden(natural,intact,score_hop,decision,eval_end)
      history=natural[onset-50:onset].astype(float); x=np.arange(len(history),dtype=float)
      slope=float(np.polyfit(x,history,1)[0]); slope=float(np.clip(slope,0,len(transition)-1))
      predicted=float(history[-1]+slope*(restore-onset))
      post,prefix=posterior_until(erased,restore,transition,BETA,RESTART); centers,_=centers_and_masses(post,score_hop)
      states,_,_,_,_,_=compare_features(erased,post,centers,transition,score_hop,restore,decision)
      selected=int(np.argmin(np.abs(np.asarray(centers)-predicted))); burdens=[]
      for state in states:
        path=natural.copy(); path[decision:]=continue_path(erased,state.copy(),transition,decision,end); burdens.append(burden(path,intact,score_hop,decision,eval_end))
      sb=burdens[selected]; ob=min([nb]+burdens); correct=int(intact[restore-1]); distance=abs(centers[selected]-correct)*score_hop
      rows.append({'corpus':corpus,'group':group,'id':identifier,'window_sec':[start,stop],
                   'persistent_event':nb>.5,'natural_burden':nb,'selector_burden':sb,'oracle_burden':ob,
                   'selector_improvement':nb-sb,'oracle_improvement':nb-ob,'anchor_distance_sec':distance,
                   'anchor_within_2sec':distance<=2.,'intact_false_outage':intact_false,'slope_state_per_frame':slope})
    return rows

def load_asap(cfg,transition):
    root=ROOT/cfg['asap_root']; manifest=list(csv.DictReader((ROOT/cfg['manifest']).open(encoding='utf-8',newline='')))
    metadata={r['midi_performance']:r for r in csv.DictReader((root/'metadata.csv').open(encoding='utf-8',newline=''))}; rows=[]
    for i,item in enumerate(manifest,1):
      meta=metadata[item['performance']]; pn=load_notes(root/meta['midi_performance']); sn=load_notes(root/meta['midi_score']); pd,sd=max(n[1] for n in pn),max(n[1] for n in sn); sh=.1*sd/pd
      base=cosine_cost(notes_to_chroma_stream(pn,60.,.1),notes_to_chroma_stream(sn,sd,sh)); rows+=evaluate(base,sh,transition,'ASAP',item['composition'],item['performance']); print(f"[ASAP {i}/{len(manifest)}] {item['performance']}")
    return rows

def load_pianocore(transition):
    cases=json.loads(ACQ.read_text())['rows']; rows=[]
    for i,case in enumerate(cases,1):
      pn=load_notes(ROOT/case['performance_path']); sn=load_notes(ROOT/case['score_path']); pd,sd=max(n[1] for n in pn),max(n[1] for n in sn); sh=.1*sd/pd
      base=cosine_cost(notes_to_chroma_stream(pn,60.,.1),notes_to_chroma_stream(sn,sd,sh)); rows+=evaluate(base,sh,transition,'PianoCoRe',case['composer'],case['id']); print(f"[PianoCoRe {i}/{len(cases)}] {case['id']}")
    return rows

def main():
    cfg=yaml.safe_load(ASAP_CFG.read_text()); temporal=yaml.safe_load(TEMPORAL.read_text()); transition=np.asarray(temporal['bayes']['transition'])
    with threadpool_limits(limits=1): rows=load_asap(cfg,transition)+load_pianocore(transition)
    summaries={}; all_gates=[]
    for corpus in ('ASAP','PianoCoRe'):
      rr=[r for r in rows if r['corpus']==corpus]; metrics={k:cluster_bootstrap(rr,k) for k in ('natural_burden','selector_burden','oracle_burden','selector_improvement','oracle_improvement')}
      nm=metrics['natural_burden']['mean']; si=metrics['selector_improvement']; oi=metrics['oracle_improvement']; recall=float(np.mean([r['anchor_within_2sec'] for r in rr])); helpful=sum(r['selector_improvement']>0 for r in rr); harmful=sum(r['selector_improvement']<0 for r in rr)
      gates={'recovery':si['mean']>=.2*nm and si['ci95'][0]>0,'fraction_of_oracle':oi['mean']>0 and si['mean']>=.5*oi['mean'],
             'direction':helpful>harmful,'anchor_recall':recall>=.8,'no_intact_false_outage':not any(r['intact_false_outage'] for r in rr)}
      summaries[corpus]={'interventions':len(rr),'events':sum(r['persistent_event'] for r in rr),'affected_groups':len(set(r['group'] for r in rr if r['persistent_event'])),
                         'anchor_recall_at_2sec':recall,'helpful':helpful,'harmful':harmful,'metrics':metrics,'gates':gates}; all_gates+=list(gates.values())
    events=sum(r['persistent_event'] for r in rows); groups=len(set((r['corpus'],r['group']) for r in rows if r['persistent_event'])); incidence=events>=30 and groups>=15; all_pass=bool(incidence and all(all_gates))
    result={'scope':'Known-outage P3-Ra opened-corpus development; not natural P3','combined_incidence':{'events':events,'affected_groups':groups,'gate':incidence},
            'corpora':summaries,'all_gate_pass':all_pass,'decision':'download_preserved_pianocore_validation' if all_pass else 'preoutage_anchor_not_supported','rows':rows}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

if __name__=='__main__': main()
