#!/usr/bin/env python3
"""Extract and preflight the two frozen MAESTRO direct-MIDI supplements."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pretty_midi

try:
    from scripts.p3_natural_restart_candidates import load_notes
except ImportError:
    from p3_natural_restart_candidates import load_notes

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/maestro-v3.0.0-midi.zip"
OUT_DIR = ROOT / "data/pianovam_p3_natural/supplement_maestro"
REPORT = ROOT / "results/p3_natural_maestro_supplement.json"
EXPECTED_ARCHIVE_SHA = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
ITEMS = [
    {"record_time": "MAESTRO_Haydn_HobXVI39_2006", "score_composer": "Haydn,_Joseph",
     "piece": "Piano Sonata Hob.XVI:39 complete", "rule_index": "MAESTRO_HAYDN_HOB_XVI_39",
     "archive_member": "maestro-v3.0.0/2006/MIDI-Unprocessed_23_R1_2006_01-05_ORIG_MID--AUDIO_23_R1_2006_02_Track02_wav.midi",
     "score_sources": ["asap_dataset/Haydn/Keyboard_Sonatas/39-1/midi_score.mid",
                       "asap_dataset/Haydn/Keyboard_Sonatas/39-2/midi_score.mid",
                       "asap_dataset/Haydn/Keyboard_Sonatas/39-3/midi_score.mid"]},
    {"record_time": "MAESTRO_Liszt_Gnomenreigen_2011", "score_composer": "Liszt,_Franz",
     "piece": "Concert Etude S.145 No.2 Gnomenreigen", "rule_index": "MAESTRO_LISZT_S145_2",
     "archive_member": "maestro-v3.0.0/2011/MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_10_Track10_wav.midi",
     "score_sources": ["asap_dataset/Liszt/Concert_Etude_S145/2/midi_score.mid"]},
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def concatenate_scores(paths, destination):
    output = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, name="concatenated score")
    offset = 0.0
    for path in paths:
        source = pretty_midi.PrettyMIDI(str(path))
        notes = [n for track in source.instruments for n in track.notes]
        start = min(n.start for n in notes)
        for note in notes:
            instrument.notes.append(pretty_midi.Note(velocity=note.velocity, pitch=note.pitch,
                start=offset + note.start-start, end=offset + note.end-start))
        offset = max(n.end for n in instrument.notes) + 1.0
    output.instruments.append(instrument); output.write(str(destination))


def cosine(a, b):
    x=np.bincount([n[2]%12 for n in a],minlength=12).astype(float)
    y=np.bincount([n[2]%12 for n in b],minlength=12).astype(float)
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))


def main():
    if sha(ARCHIVE) != EXPECTED_ARCHIVE_SHA: raise RuntimeError("MAESTRO archive hash mismatch")
    OUT_DIR.mkdir(parents=True, exist_ok=True); rows=[]
    with zipfile.ZipFile(ARCHIVE) as archive:
        names=set(archive.namelist())
        for item in ITEMS:
            if item["archive_member"] not in names: raise RuntimeError(f"missing {item['archive_member']}")
            folder=OUT_DIR/item["record_time"]; folder.mkdir(exist_ok=True)
            performance=folder/"performance.mid"; performance.write_bytes(archive.read(item["archive_member"]))
            score=folder/"score.mid"; sources=[ROOT/path for path in item["score_sources"]]
            concatenate_scores(sources, score)
            pn,sn=load_notes(performance),load_notes(score); similarity=cosine(pn,sn)
            gates={"duration_ge_90":max(n[1] for n in pn)>=90,"performance_notes_ge_200":len(pn)>=200,
                   "score_notes_ge_100":len(sn)>=100,"pitch_class_cosine_ge_0_80":similarity>=.80,
                   "performance_score_hash_distinct":sha(performance)!=sha(score)}
            rows.append({**{k:item[k] for k in ("record_time","score_composer","piece","rule_index")},
                "performance_path":str(performance.relative_to(ROOT)).replace("\\","/"),
                "score_midi_path":str(score.relative_to(ROOT)).replace("\\","/"),
                "archive_member":item["archive_member"],"score_sources":item["score_sources"],
                "score_source_sha256":[sha(p) for p in sources],"performance_sha256":sha(performance),
                "score_sha256":sha(score),"performance_duration_sec":max(n[1] for n in pn),
                "score_duration_sec":max(n[1] for n in sn),"performance_notes":len(pn),"score_notes":len(sn),
                "pitch_class_cosine":similarity,"gates":gates,"eligible":all(gates.values())})
    result={"scope":"Frozen MAESTRO direct-MIDI natural-P3 development supplement; no tracker outcomes",
        "archive_sha256":sha(ARCHIVE),"counts":{"fixed":len(rows),"eligible":sum(r['eligible'] for r in rows),
        "composers":len({r['score_composer'] for r in rows if r['eligible']})},
        "all_fixed_eligible":all(r['eligible'] for r in rows),"validation_eligible":False,"rows":rows}
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
