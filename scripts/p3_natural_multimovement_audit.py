#!/usr/bin/env python3
"""Audit the fixed Haydn complete-sonata item against ordered movement scores."""

import json
from pathlib import Path

import numpy as np

try:
    from scripts import p3_natural_coverage_units as coverage
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_coverage_units as coverage
    import p3_natural_event_restart_candidates as event

ROOT=Path(__file__).resolve().parents[1]
MAESTRO=ROOT/'results/p3_natural_maestro_supplement.json'
OUT=ROOT/'results/p3_natural_multimovement_audit.json'
SCORES=[ROOT/f'asap_dataset/Haydn/Keyboard_Sonatas/39-{i}/midi_score.mid' for i in (1,2,3)]

def cosine(notes,score):
    x=np.bincount([n[2]%12 for n in notes],minlength=12).astype(float)
    y=np.bincount([n[2]%12 for n in score],minlength=12).astype(float)
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))

def main():
    source=json.loads(MAESTRO.read_text(encoding='utf8'))
    record=next(r for r in source['rows'] if r['record_time']=='MAESTRO_Haydn_HobXVI39_2006')
    candidate=event.process(record); performance=event.load_notes(ROOT/record['performance_path'])
    pe=event.onset_events(performance,event.PERFORMANCE_ONSET_TOLERANCE)
    cuts=sorted(set([0,len(pe)]+[int(c['performance_index']) for c in candidate['candidates']]))
    structural={'exactly_two_boundaries':len(candidate['candidates'])==2,'three_segments':len(cuts)==4}
    rows=[]
    if all(structural.values()):
      for number,(start,end,score_path) in enumerate(zip(cuts,cuts[1:],SCORES),1):
        score_notes=event.load_notes(score_path); se=event.onset_events(score_notes,event.SCORE_ONSET_TOLERANCE)
        segment_events=pe[start:end]
        anchors=event.event_anchors([x['token'] for x in segment_events],[x['token'] for x in se])
        unit=coverage.unit_metrics(anchors,segment_events,len(se),0,len(segment_events))
        lo=segment_events[0]['time_sec']; hi=segment_events[-1]['latest_end_sec']
        segment_notes=[n for n in performance if lo<=n[0]<=hi]; similarity=cosine(segment_notes,score_notes)
        gates={'pitch_class_cosine_ge_0_80':similarity>=.80,**unit['gates']}
        rows.append({'movement':number,'start_sec':lo,'end_sec':hi,'pitch_class_cosine':similarity,
          'performance_notes':len(segment_notes),'score_notes':len(score_notes),'anchor_count':unit['anchor_count'],
          'score_span':unit['score_span'],'monotone_share':unit['monotone_share'],'gates':gates,
          'reviewable':all(gates.values())})
    result={'scope':'Outcome-blind ordered multi-movement normalization; no tracker/P3 outcomes',
      'record_time':record['record_time'],'candidate_boundary_times_sec':[c['time_sec'] for c in candidate['candidates']],
      'structural_gates':structural,'movements':rows,'all_movements_reviewable':len(rows)==3 and all(r['reviewable'] for r in rows),
      'validation_eligible':False}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
