#!/usr/bin/env python3
"""Extract and audit the one frozen MAESTRO Scriabin supplement."""

import hashlib,json,zipfile
from pathlib import Path
import numpy as np

try:
 from scripts import p3_natural_coverage_units as coverage
 from scripts import p3_natural_event_restart_candidates as event
except ImportError:
 import p3_natural_coverage_units as coverage
 import p3_natural_event_restart_candidates as event

ROOT=Path(__file__).resolve().parents[1]; ARCHIVE=ROOT/'data/maestro-v3.0.0-midi.zip'
OUTDIR=ROOT/'data/pianovam_p3_natural/supplement_maestro/MAESTRO_Scriabin_Sonata5_2008'
OUT=ROOT/'results/p3_natural_maestro_scriabin.json'
MEMBER='maestro-v3.0.0/2008/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--4.midi'
SCORE='asap_dataset/Scriabin/Sonatas/5/midi_score.mid'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cosine(a,b):
 x=np.bincount([n[2]%12 for n in a],minlength=12).astype(float);y=np.bincount([n[2]%12 for n in b],minlength=12).astype(float)
 return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
def main():
 OUTDIR.mkdir(parents=True,exist_ok=True); perf=OUTDIR/'performance.mid'
 with zipfile.ZipFile(ARCHIVE) as z:perf.write_bytes(z.read(MEMBER))
 score=ROOT/SCORE;pn=event.load_notes(perf);sn=event.load_notes(score);sim=cosine(pn,sn)
 gates={'duration_ge_90':max(n[1] for n in pn)>=90,'performance_notes_ge_200':len(pn)>=200,
        'score_notes_ge_100':len(sn)>=100,'pitch_class_cosine_ge_0_80':sim>=.80,
        'performance_score_hash_distinct':sha(perf)!=sha(score)}
 record={'record_time':'MAESTRO_Scriabin_Sonata5_2008','score_composer':'Scriabin,_Aleksandr',
  'piece':'Piano Sonata No.5 Op.53','rule_index':'MAESTRO_SCRIABIN_OP53',
  'performance_path':str(perf.relative_to(ROOT)).replace('\\','/'),'score_midi_path':SCORE,
  'performance_duration_sec':max(n[1] for n in pn),'score_duration_sec':max(n[1] for n in sn),
  'performance_notes':len(pn),'score_notes':len(sn),'pitch_class_cosine':sim,
  'performance_sha256':sha(perf),'score_sha256':sha(score),'gates':gates,'eligible':all(gates.values())}
 candidate=event.process(record) if record['eligible'] else None
 unit=coverage.process(record,candidate) if candidate else None
 result={'scope':'Frozen MAESTRO train direct-MIDI Scriabin supplement; no tracker/P3 outcomes',
  'archive_member':MEMBER,'record':record,'candidate_summary':None if not candidate else {
   'candidate_count':candidate['candidate_count'],'unique_anchor_count':candidate['unique_anchor_count'],
   'unique_anchor_coverage':candidate['unique_anchor_coverage'],'content_mapping_weak':candidate['content_mapping_weak']},
  'coverage':unit,'reviewable':bool(unit and unit['reviewable_units']),'validation_eligible':False}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in result.items() if k!='coverage'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
