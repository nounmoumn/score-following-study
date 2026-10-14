"""Audit annotation anchors and MIDI support for four null-with-onset episodes."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import pretty_midi
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_onset_development/null_numeric'
SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pitches(notes,t,tol):return sorted({n.pitch for n in notes if abs(n.start-t)<=tol+1e-9})
def jac(a,b):
    a,b=set(a),set(b);return len(a&b)/len(a|b) if a|b else 1.
def main():
    episodes=json.loads((BASE/'episode_summary.json').read_text())['healthy_during_null_event_rows']
    manifest={r['case_id']:r for r in read(SOURCE/'manifest.csv')}
    ap=ROOT/'asap_dataset/asap_annotations.json';annotations=json.loads(ap.read_text(encoding='utf-8'))
    rows=[];anchors=[];provenance={}
    for e in episodes:
        c=manifest[e['case_id']];score=ROOT/'asap_dataset'/c['score'];perf=ROOT/'asap_dataset'/c['performance']
        sm=pretty_midi.PrettyMIDI(str(score));pm=pretty_midi.PrettyMIDI(str(perf))
        sn=[n for i in sm.instruments for n in i.notes];pn=[n for i in pm.instruments for n in i.notes]
        ann=annotations[c['performance']];pb=np.asarray(ann['performance_beats']);sb=np.asarray(ann['midi_score_beats']);order=np.argsort(pb);pb,sb=pb[order],sb[order]
        pathfile=SOURCE/'cases'/(c['case_id']+'.npz')
        with np.load(pathfile) as z:path=z['intact_bayes'];truth=z['truth_frames'];hv=float(z['hop_score_sec'])
        onset=float(e['during_null_onset']);onframe=round(onset*10);start=int(e['start_frame']);end=int(e['end_frame_exclusive'])
        j=int(np.searchsorted(pb,onset));left=max(0,j-1);right=min(len(pb)-1,j)
        local=[]
        for side,k in [('left',left),('right',right)]:
            sp=pitches(sn,float(sb[k]),.020);pp=pitches(pn,float(pb[k]),.045)
            a=dict(case_id=c['case_id'],episode_start_sec=float(e['start_sec']),side=side,beat_index=int(k),
                performance_beat_sec=float(pb[k]),score_beat_sec=float(sb[k]),score_onset_pitches=sp,
                performance_onset_pitches=pp,score_onset_supported=bool(sp),performance_onset_supported=bool(pp),pitch_jaccard=jac(sp,pp))
            anchors.append(a);local.append(a)
        perf_onsets=[n.start for n in pn if float(e['start_sec'])<=n.start<float(e['end_sec'])]
        score_progress=float((truth[end-1]-truth[start])*hv);tracker_progress=float((path[end-1]-path[start])*hv)
        onset_score=float(truth[onframe]*hv);pred=np.interp(onset,pb,sb)
        assert abs(onset_score-pred)<1e-8
        rows.append(dict(case_id=c['case_id'],composition=c['composition'],episode_start_sec=float(e['start_sec']),episode_end_sec=float(e['end_sec']),
            onset_sec=onset,duration_sec=float(e['duration_sec']),performance_note_onsets_inside=len(perf_onsets),
            score_progress_during_null=score_progress,tracker_progress_during_null=tracker_progress,
            signed_error_change_during_null=tracker_progress-score_progress,
            annotated_score_at_onset=onset_score,tracker_score_at_onset=float(path[onframe]*hv),
            annotation_interval_perf_sec=float(pb[right]-pb[left]),annotation_interval_score_sec=float(sb[right]-sb[left]),
            onset_fraction_between_beats=float((onset-pb[left])/(pb[right]-pb[left])),
            both_annotation_endpoints_score_onset_supported=all(a['score_onset_supported'] for a in local),
            both_annotation_endpoints_performance_onset_supported=all(a['performance_onset_supported'] for a in local),
            min_endpoint_pitch_jaccard=min(a['pitch_jaccard'] for a in local),
            sustained_after_return=bool(e['sustained_after_return'])))
        for p in (score,perf,pathfile):provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=dict(scope='Retrospective four exact-null event episodes. Endpoint onset support is a consistency check, not independent GT certification.',episodes=rows,anchors=anchors,
        provenance={**provenance,**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'episode_summary.json',SOURCE/'manifest.csv',ap)}})
    (BASE/'anchor_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'episodes':rows,'anchors':anchors},indent=2))
if __name__=='__main__':main()
