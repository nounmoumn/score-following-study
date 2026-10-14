"""Development structural audit of two existing score variants against SMD."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pretty_midi
from p3_smd_alignment_pilot import ROOT, events, align

PERF=ROOT/'data/smd_direct_midi/Beethoven_Op031No2-01_002_20090916-SMD.mid'
OUT=ROOT/'results/p3_external_metadata/smd_beethoven_repeat_audit.json'

def load(path, threshold=.045):
    midi=pretty_midi.PrettyMIDI(str(path))
    notes=[{'time':note.start,'pitch':note.pitch} for inst in midi.instruments for note in inst.notes]
    return events(notes,'time',threshold=threshold),len(notes),float(midi.get_end_time())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-alignment',action='store_true')
    args=parser.parse_args()
    pe,pn,pdur=load(PERF)
    result={'performance':str(PERF.relative_to(ROOT)),'performance_notes':pn,
            'performance_events':len(pe),'performance_duration_sec':pdur,
            'performance_sha256':hashlib.sha256(PERF.read_bytes()).hexdigest(),'variants':[],
            'caveat':'Exploratory score-structure development audit only; no P3 outcomes or independent GT.'}
    for name in ('17-1','17-1_no_repeat'):
        score=ROOT/f'asap_dataset/Beethoven/Piano_Sonatas/{name}/midi_score.mid'
        se,sn,sdur=load(score,threshold=1e-4)
        row={'variant':name,'score_notes':sn,'score_events':len(se),'score_duration_sec':sdur,
             'score_sha256':hashlib.sha256(score.read_bytes()).hexdigest()}
        print(json.dumps(row),flush=True)
        if args.run_alignment:
            cost,mapping,steps=align(se,pe)
            similarities=[len(se[mapping[j]][1]&pe[j][1])/len(se[mapping[j]][1]|pe[j][1]) for j in mapping]
            row.update({'mean_path_cost':float(cost),'path_steps':steps,
                        'score_events_covered':len(set(mapping.values())),
                        'low_pitch_agreement_events':sum(x<.5 for x in similarities),
                        'median_pitch_jaccard':float(np.median(similarities))})
            print(json.dumps(row),flush=True)
        result['variants'].append(row)
    if args.run_alignment:
        result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['align_script_sha256']=hashlib.sha256(Path(__file__).with_name('p3_smd_alignment_pilot.py').read_bytes()).hexdigest()
        OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
