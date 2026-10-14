"""Descriptive exact-null context on unchanged healthy landmarks, no fitted rule."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_onset_development'
def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    src=BASE/'null_numeric';frames=read(src/'frames.csv');marks=read(src/'landmarks.csv')
    lookup={(r['case_id'],int(r['frame'])):r for r in frames};result=[]
    for r in marks:
        end=round(float(r['time_sec'])*10)
        ff=[lookup[(r['case_id'],i)]['flat']=='True' for i in range(end-20,end)]
        any_flat=any(ff);two=any(a and b for a,b in zip(ff,ff[1:]))
        result.append(dict(case_id=r['case_id'],time_sec=r['time_sec'],event=r['event']=='True',
            any_flat=any_flat,two_flat=two,all_flat=all(ff),flat_count=sum(ff)))
    summary={'scope':'Exact-null descriptive context only. Same396healthy landmarks, prior2sec and future2sec. No new validation gate, no causal claim.', 'rules':{}}
    for field in ('any_flat','two_flat','all_flat'):
        tp=sum(r[field] and r['event'] for r in result);fp=sum(r[field] and not r['event'] for r in result)
        fn=sum(not r[field] and r['event'] for r in result);tn=sum(not r[field] and not r['event'] for r in result)
        summary['rules'][field]=dict(tp=tp,fp=fp,fn=fn,tn=tn,event_rate_with=tp/(tp+fp) if tp+fp else None,event_rate_without=fn/(fn+tn) if fn+tn else None)
    summary['flat_frames_with_nonzero_input']=sum(r['flat']=='True' and r['zero_input']!='True' for r in frames)
    summary['provenance']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),src/'frames.csv',src/'landmarks.csv')}
    (src/'context.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
