"""Outcome-blind full-development review requests; never marks anchors approved."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    src=ROOT/'results/p3_temporal_20260918_000058/manifest.csv'
    with src.open(encoding='utf-8',newline='') as f:cases=list(csv.DictReader(f))
    out=ROOT/'results/p3_onset_development/blind_review';out.mkdir(exist_ok=True)
    rows=[]
    for c in cases:
        score=ROOT/'asap_dataset'/c['score'];perf=ROOT/'asap_dataset'/c['performance']
        xml=score.with_name('xml_score.musicxml')
        rows.append(dict(review_id='review_'+c['case_id'],performance=str(perf),score_midi=str(score),
            score_xml=str(xml) if xml.exists() else '',performance_sha256=hashlib.sha256(perf.read_bytes()).hexdigest(),
            score_sha256=hashlib.sha256(score.read_bytes()).hexdigest(),
            review_start_sec=17.9,review_end_sec=82.0,
            instruction='Independently map note onsets and repeat occurrences in this complete interval. Record score quarter positions, note IDs, and unresolved spans. Do not infer silent internal beats as observed anchors.'))
    with (out/'requests.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=['review_id','performance_note_id','performance_sec','score_note_id','score_quarter','repeat_occurrence',
            'anchor_kind','time_lower_sec','time_upper_sec','reviewer_id','reviewed_at','evidence','status']
    for reviewer in ('A','B'):
        target=out/f'reviewer_{reviewer}_anchors.csv'
        if not target.exists():
            with target.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerow(fields)
    meta=dict(requests=len(rows),annotation_complete=0,selection='All 40 existing development performances, full union of audited windows. No outcome-based sampling.',
              review_duration_total_minutes=len(rows)*(82-17.9)/60,
              caveat='This is source/annotation request material, not a rendered annotation interface. Reviewer files deliberately empty. No independent annotations obtained.',
              source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# 독립 검토 요청 — 완료 주석 0

requests.csv만 검토자에게 제공한다. gt_dependency의 정답/오차/사건 자료와 기존 오류 그림은 첫 독립 판단 때 제공하지 않는다. 요청은 전체 개발40연주에서 동일17.9–82초 범위이며 사건에 따라 선택하지 않았다. 검토자A/B가 서로의 판정을 보지 않고 각자의 앵커 파일을 작성한 후에만 차이를 대조한다. 현재 파일은 빈 양식이다.

원악보와 MIDI note ID로 타건 대응·반복 회차를 기록하고 지속음 내부는 관측 앵커로 만들지 않는다. 필요하면 앞뒤 문맥을 원파일에서 확인한다. MusicXML 위치와 score MIDI 시계 변환도 별도 검증해야 한다. 이 패킷은 검토 UI 또는 사람이 검토한 정답이 아니다. 실제 검토자·검토 시간을 확보하기 전 완료로 바꾸지 않는다.
''',encoding='utf-8')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
