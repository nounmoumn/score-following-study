"""Build a static error review packet. No automatic GT acceptance."""
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pretty_midi

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_external_metadata'
OUT=BASE/'anchor_review'

def notes(path):
    m=pretty_midi.PrettyMIDI(str(path))
    return [(n.start,n.end,n.pitch) for inst in m.instruments for n in inst.notes]

def main():
    OUT.mkdir(exist_ok=True)
    review_csv=OUT/'development_error_reviews.csv'
    prior={}
    if review_csv.exists():
        with review_csv.open(encoding='utf-8',newline='') as f:
            prior={r['review_id']:r for r in csv.DictReader(f)}
    data=json.loads((BASE/'beethoven_repeat_calibration.json').read_text())
    rows=[]; cards=[]; provenance={}
    for case in data['cases']:
        name=case['name']; folder=ROOT/f"asap_dataset/Beethoven/Piano_Sonatas/{case['variant']}"
        mapping_path=BASE/f'{name}_alignment_mapping.npz'
        mapping=np.load(mapping_path)
        times=mapping['performance_onset_sec']; positions=mapping['mapped_score_onset_sec']
        pn=notes(folder/f'{name}.mid'); sn=notes(folder/'midi_score.mid')
        beats=case['beat_rows']
        for k,r in enumerate(beats):
            if abs(r['error_score_sec'])<=.5: continue
            key=f'{name}_beat{k:04d}'; t=r['performance_sec']; s=r['true_score_sec']
            fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
            idx=(times>=t-4)&(times<=t+4)
            # Include bracketing points even for long gaps.
            lo=max(0,int(np.searchsorted(times,t-4))-1);hi=min(len(times),int(np.searchsorted(times,t+4))+1)
            axes[0].plot(times[lo:hi],positions[lo:hi],'.-',label='Frozen DTW onset map + linear interpolation')
            near=[x for x in beats if abs(x['performance_sec']-t)<=4]
            axes[0].plot([x['performance_sec'] for x in near],[x['true_score_sec'] for x in near],'x',color='green',label='Published ASAP beat annotation')
            axes[0].axvline(t,color='red',alpha=.6);axes[0].set_xlim(t-4,t+4)
            axes[0].set_ylabel('Score seconds');axes[0].legend(fontsize=7)
            for ax,ns,center,label in ((axes[1],pn,t,'Performance seconds'),(axes[2],sn,s,'Score seconds')):
                for a,b,p in ns:
                    if a<=center+4 and b>=center-4:
                        ax.plot([max(a,center-4),min(b,center+4)],[p,p],linewidth=3)
                ax.axvline(center,color='red',alpha=.6);ax.set_xlim(center-4,center+4)
                ax.set_ylabel('MIDI pitch');ax.set_xlabel(label)
            fig.suptitle(f"{key}: error {r['error_score_sec']:+.3f} score seconds — review pending")
            image=OUT/f'{key}.png';fig.savefig(image,dpi=130);plt.close(fig)
            j=int(np.searchsorted(times,t));left=max(0,j-1);right=min(len(times)-1,j)
            row={'review_id':key,'performance_sec':t,'annotated_score_sec':s,
                 'automatic_score_sec':r['predicted_score_sec'],
                 'left_onset_sec':float(times[left]),'right_onset_sec':float(times[right]),
                 'left_mapped_score_sec':float(positions[left]),'right_mapped_score_sec':float(positions[right]),
                 'review_status':'pending','reviewer':'','judgment':'','notes':''}
            if key in prior:
                for field in ('review_status','reviewer','judgment','notes'):
                    row[field]=prior[key][field]
            rows.append(row)
            cards.append(f'<section><h2>{html.escape(key)}</h2><img src="{image.name}" alt="Alignment and note timelines"><p>Review pending. Check onset matching, annotation interpretation, and interpolation separately.</p></section>')
        for path in (mapping_path,folder/'midi_score.mid',folder/f'{name}.mid'):
            provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with review_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P3 alignment review</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}section{border-top:1px solid #ccc;margin-top:30px}</style><h1>Development alignment error review</h1><p>Eight selected errors, not a representative quality estimate. Green beat annotations belong only to the named ASAP performance, never to SMD. All reviewer decisions remain pending.</p>'+''.join(cards),encoding='utf-8')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # Candidate locations are only prompts for independent annotation, never accepted anchors.
    pilot=json.loads((BASE/'smd_bwv875_alignment_pilot.json').read_text())
    template=OUT/'smd_bwv875_anchor_requests.csv'
    if not template.exists():
        with template.open('w',newline='',encoding='utf-8') as f:
            fields=['performance_sec','automatic_score_quarter_hint','confirmed_score_quarter',
                    'uncertainty_sec','reviewer','evidence','status']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in pilot['low_pitch_agreement_events']:
                writer.writerow({'performance_sec':r['performance_sec'],
                                 'automatic_score_quarter_hint':r['score_quarter'],
                                 'status':'pending'})
    print(f'Created {len(rows)} review cards and pending CSV at {OUT}')

if __name__=='__main__':main()
