"""Outcome-blind review packet for two persistent exact-null episodes."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_onset_development/null_numeric';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
CASES=[('n019',51.7,55.5),('n026',28.5,29.4)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    manifest={r['case_id']:r for r in read(SOURCE/'manifest.csv')};ap=ROOT/'asap_dataset/asap_annotations.json';ann=json.loads(ap.read_text(encoding='utf-8'))
    out=BASE/'review_packet';out.mkdir(exist_ok=True);requests=[];prov={}
    for cid,start,end in CASES:
        c=manifest[cid];perf=ROOT/'asap_dataset'/c['performance'];score=ROOT/'asap_dataset'/c['score']
        pm=pretty_midi.PrettyMIDI(str(perf));sm=pretty_midi.PrettyMIDI(str(score));a=ann[c['performance']]
        pb=np.asarray(a['performance_beats']);sb=np.asarray(a['midi_score_beats']);order=np.argsort(pb);pb,sb=pb[order],sb[order]
        lo=max(0,start-3);hi=end+3;j0=max(0,np.searchsorted(pb,lo)-1);j1=min(len(pb),np.searchsorted(pb,hi)+1)
        fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
        for ax,midi,x0,x1,label in [(axes[0],pm,lo,hi,'performance seconds'),(axes[1],sm,float(np.interp(lo,pb,sb)),float(np.interp(hi,pb,sb)),'score seconds')]:
            for inst in midi.instruments:
                for n in inst.notes:
                    if n.end>=x0 and n.start<=x1:ax.plot([max(n.start,x0),min(n.end,x1)],[n.pitch,n.pitch],lw=2)
            ax.set_xlim(x0,x1);ax.set_ylabel('MIDI pitch');ax.set_xlabel(label)
        axes[0].axvspan(start,end,color='orange',alpha=.2,label='candidate silent interval');axes[0].legend()
        for k in range(j0,j1):
            axes[0].axvline(pb[k],color='green',alpha=.3);axes[1].axvline(sb[k],color='green',alpha=.3)
        fig.suptitle(f'{cid} {c["composition"]} — outcome-hidden alignment review')
        image=out/(cid+'.png');fig.savefig(image,dpi=140);plt.close(fig)
        requests.append(dict(review_id='null_'+cid,case_id=cid,composition=c['composition'],performance=str(perf.relative_to(ROOT)),
            score_midi=str(score.relative_to(ROOT)),score_xml=str(score.with_name('xml_score.musicxml').relative_to(ROOT)),
            candidate_start_sec=start,candidate_end_sec=end,image=image.name,
            question='Identify the performed score positions immediately before and after the orange interval, repeat occurrence, and whether linear beat timing inside is defensible. Do not inspect tracker output.',
            reviewer_a_status='pending',reviewer_b_status='pending'))
        for p in (perf,score,score.with_name('xml_score.musicxml'),image):prov[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with (out/'requests.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(requests[0]));w.writeheader();w.writerows(requests)
    fields=['review_id','reviewer_id','before_performance_note_ids','before_score_note_ids','after_performance_note_ids','after_score_note_ids','repeat_occurrence','score_position_before','score_position_after','interior_time_lower','interior_time_upper','judgment','evidence','reviewed_at','status']
    for name in ('reviewer_A.csv','reviewer_B.csv'):
        p=out/name
        if not p.exists():
            with p.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(fields)
    meta=dict(requests=2,completed_reviews=0,outcome_hidden=True,tracker_artifacts_excluded=True,
        caveat='PNG and source paths are review aids, not human annotations. Green lines are published beat annotations; orange is selected from prior automated outcome audit.',
        provenance={**prov,str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),str(ap.relative_to(ROOT)):hashlib.sha256(ap.read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
