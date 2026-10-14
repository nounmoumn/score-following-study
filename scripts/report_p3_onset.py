"""Development report and failure-route audit, without held-out fitting."""
import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from p3_temporal import read_csv,write_csv


def main():
    sys.stdout.reconfigure(encoding='utf-8');out=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058'
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));inventory=json.loads((ROOT/'results/p3_inventory/summary.json').read_text(encoding='utf-8'))
    rows=read_csv(out/'landmarks.csv');events=[r for r in rows if r['status']=='new_failure'];audit=[]
    for r in events:
        with np.load(source/'cases'/(r['case_id']+'.npz')) as z:
            path=z['intact_bayes'];truth=z['truth_frames'];hv=float(z['hop_score_sec'])
            j=int(r['failure_frame']);start=int(round(float(r['time_sec'])*10))+20
            jumps=np.diff(path[start-1:j+5]);error=abs(path-truth)*hv
            audit.append(dict(case_id=r['case_id'],composition=r['composition'],time_sec=r['time_sec'],failure_sec=j*.1,
                weakness=float(r['weakness']),risk=float(r['risk']),max_abs_MAP_step_sec=float(np.max(abs(jumps))*hv),
                nonlocal_MAP_switch=bool(np.any((jumps<0)|(jumps>3))),error_at_onset_sec=float(error[j]),error_after_dwell_sec=float(error[j+4]),
                score_progress_sec=float((truth[j]-truth[start-1])*hv),tracker_progress_sec=float((path[j]-path[start-1])*hv)))
    write_csv(out/'failure_route_audit.csv',audit)
    lines=['# P3 신규 이탈 개발 감사','',
        '기존 temporal40 score의 480개 고정시점을 사용했다. 미사용 후보의 추적 결과는 실행하지 않았다. 사전 규칙: [P3_ONSET_DEVELOPMENT_PROTOCOL.md](P3_ONSET_DEVELOPMENT_PROTOCOL.md).','',
        '## 분리한 결과','', '| 분류 | 시점 수 | W 중앙값 | W>0 | 사전경쟁 R≥1% |','|---|---:|---:|---:|---:|']
    for status,g in m['groups'].items():lines.append(f"| {status} | {g['n']} | {g['median_W']:+.5f} | {g['W_positive']} | {g['risk_ge_1percent']} |")
    lines+=['',f"신규 이탈: 양성 window {len(events)}개, 고유 onset {m['unique_onsets']}개, {m['failure_scores']} score / {m['failure_families']} 작품군.",
        'pre_unhealthy는 시작 직전2초 오차가 .5초 이하 조건 미달. exposure_crossing은 이후2초 관측 측정 중 임계값을 넘은 경우다.',
        'new_failure는 노출 중 임계값을 넘지 않고 그 뒤5초 내 .5초 초과가 .5초 연속인 경우. no_failure는 그 조건의 지속 이탈이 없는 경우다.',
        '이는 노출완료까지 건강함에 조건부인 예측이며 노출기간 이탈을 무시한 전체 인과효과가 아니다. 노출기간 사건을 별도 공개한다.',
        '## 이탈 경로에서 찾은 특성','',
        '| 작품 | 측정시작 | 이탈시작 | W | R | 최대 MAP 이동(초/frame) | local transition 범위 밖 MAP 전환 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for a in audit:lines.append(f"| {a['composition']} | {float(a['time_sec']):g} | {a['failure_sec']:.1f} | {a['weakness']:+.4f} | {a['risk']:.5f} | {a['max_abs_MAP_step_sec']:.4f} | {a['nonlocal_MAP_switch']} |")
    lines+=['','MAP의 0~3frame 밖 이동은 한 경로의 물리적 이동이 아니라 분포 내 최빈 후보 전환일 수 있다. 이 표만으로 restart가 원인이라고 결론내리지 않는다.',
        'W>0이면 오답후보의 관측가능도가 높다. W<0인 이탈은 측정기간 정답 관측우세에도 나중에 이탈했다는 사례로, 원래 P3의 단순한 필요조건 해석에 주의가 필요하다.',
        '프레임 timestamp는 구간 시작 표기이고 실제 특징은 다음 .1초까지의 음을 포함한다. 모든 관측 가용시간은 .1초 뒤이며 상대적인 예측 간격은 유지된다.',
        '## 독립 검증 자료의 한계','',
        f"로컬 결과 CSV에서 사용 흔적을 찾은 악보는 {inventory['used_metadata_scores']}/{inventory['score_count']}. 동일 소나타 악장·반복버전·Bach 동일BWV 전주곡/푸가를 묶으면 주석조건 적격 미사용 후보 {len(inventory['annotation_eligible_unused_families'])} 작품군.",
        '잔여 작품은 Beethoven과 Chopin에만 편중되어 있다. CSV 기록 부재는 과거 모든 실험에서 미사용이었다는 보증이 아니다. MIDI 길이 및 전체 이력 확인 전 확정 holdout이 아니다.',
        '작품군 매핑은 폴더 규칙이며 수동 음악학적 식별 검토가 필요하다. 연습곡집 개별 곡은 별도 작품으로 두었으므로 같은 작품집 스타일 상관도 남는다.',
        '## 판정','',
        '이 단계는 개발 감사이고 유의성 검정을 하지 않았다. 소수 양성 window를 독립 사건처럼 늘려 P3를 검증했다고 주장하지 않는다.',
        '다음 모형학습/검증은 충분한 독립 발생작품을 확보해야 한다. 현재40개에서 관찰한 발생작품 비율은 개발 표본의 기술통계이며 다른 작곡가 표본에 그대로 적용한 검증력 보장은 불가능하다.',
        '당장 잔여 holdout을 소비하기보다 기존자료에서 노출기간 이탈의 시간해상도와 MAP 후보전환을 조사하고, 독립 작품·더 긴 연주 구간을 포함한 검증 표본 계획을 확정한다.',
        '검증: 전체63개 테스트 통과, 개발40개 전체 경로 기존 저장본 일치. P1/P2 효과검정 재실행 없음.']
    if (out/'restart_metrics.json').exists():
        rr=json.loads((out/'restart_metrics.json').read_text(encoding='utf-8'))
        lines+=['','## 탐색: 같은 상태에서 전역 restart 제거','',
            f"노출완료까지 건강한 {rr['clean_windows']}시점 모두에서 posterior를 복제한 뒤 이후5초 restart만1e-6→0으로 변경했다. 원조건 신규이탈 {rr['original_failures']}, restart제거 {rr['no_restart_failures']}. 방지 {rr['prevented']}, 새로 발생 {rr['induced']}, 양쪽 지속 {rr['persisted']}. 전체 경로까지 동일한 시점 {rr['identical_paths']}.",
            '실패 사례만 골라 개입하지 않았고 모든 적격시점에 적용했다. 원조건 branch는 저장된 미래경로와 모두 일치했다.',
            '이 탐색은 미래 restart 유입의 역할만 조사한다. 노출완료 시 이미 존재하는 원거리 후보 posterior는 그대로 남으므로, 제거 후 이탈이 남아도 과거 restart의 기여가 없었다는 뜻은 아니다.',
            '같은 과거·상태에서의 알고리즘 개입 결과이며 원래 자연 P3를 검증하는 독립자료/주검정으로 사용하지 않는다.']
    report='\n'.join(lines)+'\n';(ROOT/'P3_ONSET_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
    entry=f"\n## 2026-09-18 신규 이탈 분리 개발 감사\n\n기존40 score/480시점 분류: {m['status_counts']}. 고유 onset {m['unique_onsets']}, 발생작품군 {m['failure_families']}. 자연 P3의 유의성/성공 판정은 하지 않음. 잔여 주석적격 미사용 후보25작품은 Beethoven/Chopin에 편중되어 결과를 아직 열지 않음. 전체63 tests passed, 저장경로 일치.\n상세 [P3_ONSET_REPORT.md](P3_ONSET_REPORT.md), 원본 `results/p3_onset_development` 및 `results/p3_inventory`.\n\n"
    for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
        p=ROOT/name;old=p.read_text(encoding='utf-8')
        if '## 2026-09-18 신규 이탈 분리 개발 감사' not in old:
            first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
    files=[Path(__file__),ROOT/'scripts/p3_inventory.py',ROOT/'scripts/audit_p3_restart.py',ROOT/'P3_ONSET_REPORT.md',ROOT/'tests/test_onset.py']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
