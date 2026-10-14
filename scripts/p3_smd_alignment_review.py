"""Audit XML/MIDI agreement and calibrate frozen pilot alignment on annotated development files."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pretty_midi
import partitura as pt
import mido
from p3_smd_alignment_pilot import ROOT, SCORE, PERF, events, align

BASE = SCORE.parent
OUT = ROOT / 'results/p3_external_metadata/smd_alignment_review.json'

def midi_events(path, quarters=False):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [{'pitch': note.pitch, 'time': (midi.time_to_tick(note.start) / midi.resolution
              if quarters else note.start)} for inst in midi.instruments for note in inst.notes]
    return events(notes, 'time', threshold=1e-4 if quarters else .045)

def main():
    xml = pt.load_musicxml(SCORE)
    se = events(xml.note_array(), 'onset_quarter')
    me = midi_events(BASE / 'midi_score.mid', quarters=True)
    md = {round(t, 4): pitches for t, pitches in me}
    mismatch = [{'quarter': t, 'xml': sorted(p), 'midi': sorted(md.get(round(t,4), set()))}
                for t,p in se if p != md.get(round(t,4), set())]
    pilot = json.loads((ROOT/'results/p3_external_metadata/smd_bwv875_alignment_pilot.json').read_text())
    low = pilot['low_pitch_agreement_events']
    tree = ET.parse(SCORE)
    marked = []
    for measure in tree.findall('.//measure'):
        ornaments = [node.tag for node in measure.findall('.//ornaments/*')]
        octaves = [node.attrib for node in measure.findall('.//octave-shift')]
        if ornaments or octaves:
            marked.append({'measure': measure.get('number'), 'ornaments': ornaments, 'octave_shifts': octaves})
    part = xml.parts[0]
    ornament_notes = [{'quarter':float(part.quarter_map(note.start.t)),
                       'pitch':int(note.midi_pitch),'ornaments':list(note.ornaments)}
                      for note in part.notes_tied if note.ornaments]
    clock_audit = {}
    for path in (BASE/'midi_score.mid', BASE/'Ahfat01M.mid', BASE/'CaoJ01M.mid', PERF):
        midi = mido.MidiFile(path)
        tempo = []
        for ti, track in enumerate(midi.tracks):
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == 'set_tempo':
                    tempo.append({'track':ti,'tick':tick,'tempo':msg.tempo})
        clock_audit[path.name] = tempo
    pe_smd = events(pt.load_performance_midi(PERF).note_array(), 'onset_sec', threshold=.045)
    _, mapping_exact, _ = align(se, pe_smd)
    _, mapping_chroma, _ = align(se, pe_smd, chroma=True)
    maximum_difference = max(abs(se[mapping_exact[j]][0]-se[mapping_chroma[j]][0]) for j in mapping_exact)
    score_seconds = midi_events(BASE/'midi_score.mid')
    truth_score = np.loadtxt(BASE/'midi_score_annotations.txt', usecols=0)
    calibration = []
    for name in ('Ahfat01M', 'CaoJ01M'):
        path = BASE/f'{name}.mid'
        pe = midi_events(path)
        truth_perf = np.loadtxt(BASE/f'{name}_annotations.txt', usecols=0)
        if len(truth_perf) != len(truth_score):
            raise ValueError('beat annotations differ in length')
        _, mapping, _ = align(score_seconds, pe)
        times = np.array([pe[j][0] for j in sorted(mapping)])
        positions = np.array([score_seconds[mapping[j]][0] for j in sorted(mapping)])
        valid = (truth_perf >= times[0]) & (truth_perf <= times[-1])
        residual = np.interp(truth_perf[valid], times, positions) - truth_score[valid]
        calibration.append({'name':name,'evaluated_beats':int(valid.sum()),
             'outside_onset_support':int((~valid).sum()),
             'median_abs_score_sec':float(np.median(abs(residual))),
             'p95_abs_score_sec':float(np.percentile(abs(residual),95)),
             'max_abs_score_sec':float(max(abs(residual))),
             'errors_gt_0_1_score_sec':int(sum(abs(residual)>.1)),
             'errors_gt_0_5_score_sec':int(sum(abs(residual)>.5)),
             'residuals':residual.tolist()})
    review = []
    for row in low:
        x, p = set(row['score_pitches']), set(row['performance_pitches'])
        j = min(range(len(pe_smd)),key=lambda j:abs(pe_smd[j][0]-row['performance_sec']))
        review.append({**row, 'same_pitch_classes': {a%12 for a in x} == {a%12 for a in p},
                       'performance_subset_of_score': p <= x,
                       'score_ornament_at_onset':any(abs(n['quarter']-row['score_quarter'])<1e-4 for n in ornament_notes),
                       'neighbor_events':[{'time':pe_smd[k][0],'pitches':sorted(pe_smd[k][1]),
                                           'mapped_quarter':se[mapping_exact[k]][0]}
                                          for k in range(max(0,j-1),min(len(pe_smd),j+2))]})
    inputs=[SCORE, PERF, BASE/'midi_score.mid', Path(__file__),
            Path(__file__).with_name('p3_smd_alignment_pilot.py'),
            BASE/'midi_score_annotations.txt']
    for name in ('Ahfat01M','CaoJ01M'):
        inputs.extend([BASE/f'{name}.mid', BASE/f'{name}_annotations.txt'])
    result={'xml_midi_onset_pitch_disagreements':mismatch,
            'xml_events':len(se),'midi_events':len(me),'score_notation_marks':marked,
            'ornament_notes':ornament_notes,'tempo_events':clock_audit,
            'smd_max_method_difference_quarters':maximum_difference,
            'low_agreement_review':review,'annotated_development_calibration':calibration,
            'provenance':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'versions':{name:importlib.metadata.version(name) for name in ('partitura','pretty-midi','numpy','mido')},
            'limitation':'ASAP calibration is development and beat-only. It cannot certify SMD note/onset GT. No tracker outcomes computed.'}
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'xml_midi_disagreements':len(mismatch),'notation_marks':marked,
                     'ornament_notes':ornament_notes,'tempo_events':clock_audit,
                     'smd_max_method_difference_quarters':maximum_difference,
                     'low_events':len(review),'octave_equivalent':sum(x['same_pitch_classes'] for x in review),
                     'subset':sum(x['performance_subset_of_score'] for x in review),
                     'calibration':[{k:v for k,v in x.items() if k!='residuals'} for x in calibration]},indent=2))

if __name__=='__main__': main()
