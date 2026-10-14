"""Read-only localization of frozen calibration errors without fixing thresholds."""
import json
import hashlib
from pathlib import Path
import numpy as np
import mido
from p3_smd_alignment_pilot import ROOT
from p3_smd_repeat_audit import load

def main():
    source=ROOT/'results/p3_external_metadata/beethoven_repeat_calibration.json'
    data=json.loads(source.read_text())
    result={'calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':[]}
    for case in data['cases']:
        base=ROOT/f"asap_dataset/Beethoven/Piano_Sonatas/{case['variant']}"
        paths=[base/'midi_score.mid',base/f"{case['name']}.mid"]
        se,_,_=load(paths[0],threshold=1e-4)
        pe,_,_=load(paths[1])
        st=np.array([x[0] for x in se]);pt=np.array([x[0] for x in pe])
        rows=[]
        for row in case['beat_rows']:
            if abs(row['error_score_sec'])<=.5: continue
            t,s=row['performance_sec'],row['true_score_sec']
            j=int(np.searchsorted(pt,t)); i=int(np.searchsorted(st,s))
            rows.append({**row,'nearest_performance_onset_sec':float(min(abs(pt-t))),
                         'nearest_score_onset_sec':float(min(abs(st-s))),
                         'bracketing_performance_onset_gap_sec':float(pt[j]-pt[j-1]) if 0<j<len(pt) else None,
                         'performance_neighbors':[{'sec':a,'pitches':sorted(b)} for a,b in pe[max(0,j-2):j+2]],
                         'score_neighbors':[{'sec':a,'pitches':sorted(b)} for a,b in se[max(0,i-2):i+2]]})
        clocks={}
        for path in paths:
            tempo=[]
            for ti,track in enumerate(mido.MidiFile(path).tracks):
                tick=0
                for msg in track:
                    tick+=msg.time
                    if msg.type=='set_tempo': tempo.append({'track':ti,'tick':tick,'tempo':msg.tempo})
            clocks[path.name]=tempo
        result['cases'].append({'name':case['name'],'errors':rows,'tempo_events':clocks})
    (source.parent/'beethoven_repeat_errors.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{ 'name':c['name'],'errors':[{k:v for k,v in r.items() if 'neighbors' not in k} for r in c['errors']],
                       'nonzero_track_tempos':sum(x['track']!=0 for ev in c['tempo_events'].values() for x in ev)} for c in result['cases']],indent=2))

if __name__=='__main__': main()
