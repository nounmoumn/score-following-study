"""Reconstruct prior-fixed observation evidence; no P3 significance testing."""
import argparse,csv,json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior,regional_evidence

def read(path):
    with open(path,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(path,rows):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);args=ap.parse_args()
    source=ROOT/args.result;out=source/'observation_audit';out.mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes']
    manifest=read(source/'manifest.csv');events=read(source/'natural_event_audit.csv');pairs=read(source/'natural_pairs.csv')
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    results=[];all_rows=[];max_residual=0.
    with threadpool_limits(limits=1):
        for case in manifest:
            query={float(r['event_time_sec']):'event' for r in events if r['case_id']==case['case_id']}
            for r in pairs:
                if r['case_id']==case['case_id']:query.setdefault(float(r['control_sec']),'control')
            if not query:continue
            u,v,hv,time,truth=load_case(case,cfg,annot);cost=cosine_cost(u,v)
            p=np.zeros(len(v));p[0]=1;path=[];rows=[]
            for i,row in enumerate(cost):
                if i:
                    p=predict_prior(p,b['transition'],b['restart_probability'])
                    p,metrics=regional_evidence(p,row,int(round(truth[i])),max(1,int(round(.5/hv))),
                        max(1,int(round(2/hv))),b['emission_beta'])
                    max_residual=max(max_residual,abs(metrics['identity_residual']))
                    item=dict(case_id=case['case_id'],frame=i,time_sec=float((i+1)*cfg['hop_sec']),**metrics)
                    rows.append(item)
                else:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                path.append(int(np.argmax(p)))
            with np.load(source/'cases'/(case['case_id']+'.npz')) as saved:
                np.testing.assert_array_equal(path,saved['intact_bayes'])
            for t,role in sorted(query.items()):
                i=int(round(t/cfg['hop_sec']))-1;current=rows[i-1]
                past=rows[i-20:i] # row zero represents frame1, ends at frame i
                assert len(past)==20 and past[-1]['frame']==i
                results.append(dict(case_id=case['case_id'],composition=case['composition'],time_sec=t,role=role,
                    evidence_now=current['observation_log_bayes_factor'],
                    evidence_past2_mean=float(np.mean([r['observation_log_bayes_factor'] for r in past])),
                    wrong_prior_mass=current['wrong_prior_mass'],correct_prior_mass=current['correct_prior_mass'],
                    prior_log_odds=current['prior_log_odds'],posterior_log_odds=current['posterior_log_odds'],
                    wrong_offset_score_sec=float((current['wrong_center']-round(truth[i]))*hv)))
            all_rows.extend(rows)
            print(case['case_id']+' complete',flush=True)
    write(out/'frames.csv',all_rows);write(out/'query_results.csv',results)
    summary=dict(max_identity_residual=max_residual,frames=len(all_rows),roles={})
    for role in ['event','control']:
        rr=[r for r in results if r['role']==role]
        summary['roles'][role]=dict(n=len(rr),
            median_log_evidence_now=float(np.median([r['evidence_now'] for r in rr])),
            median_log_evidence_past2=float(np.median([r['evidence_past2_mean'] for r in rr])),
            median_wrong_prior_mass=float(np.median([r['wrong_prior_mass'] for r in rr])),
            median_correct_prior_mass=float(np.median([r['correct_prior_mass'] for r in rr])),
            wrong_prior_ge_one_percent=sum(r['wrong_prior_mass']>=.01 for r in rr),
            evidence_now_favors_wrong=sum(r['evidence_now']<0 for r in rr),
            prior_competitor_and_wrong_evidence=sum(r['wrong_prior_mass']>=.01 and r['evidence_now']<0 for r in rr))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    sources=[Path(__file__),ROOT/'src/score_following/observation_evidence.py',ROOT/'P3_OBSERVATION_PROTOCOL.md',source/'config.yaml']
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},indent=2),encoding='utf-8')
    for f in sources:shutil.copy2(f,out/f.name)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
