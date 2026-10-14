"""Summarize scheduled P3 work and retain a reproducible stop/go record."""
import json,hashlib,shutil,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from p3_temporal import read_csv

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    root=ROOT/'results/p3_onset_development';alarm=json.loads((root/'short_signal/metrics.json').read_text(encoding='utf-8'))
    preflight=json.loads((root/'holdout_preflight/metrics.json').read_text(encoding='utf-8'))
    variance=json.loads((root/'sample_variance/metrics.json').read_text(encoding='utf-8'))
    rows=read_csv(root/'short_signal/landmarks.csv');families=defaultdict(list)
    for r in rows:families[r['work_family']].append(r)
    family_summary={}
    for rule in ['any_positive','two_consecutive']:
        event_families={g for g,rr in families.items() if any(r['sustained']=='True' for r in rr)}
        alerted={g for g,rr in families.items() if any(r[rule]=='True' for r in rr)}
        family_summary[rule]=dict(families=len(families),event_families=len(event_families),alerted_event_families=len(alerted&event_families),
            alerted_no_event_families=len(alerted-event_families),no_event_families=len(families)-len(event_families))
    (root/'short_signal/family_summary.json').write_text(json.dumps(family_summary,indent=2),encoding='utf-8')
    lines=['# P3 예약 실행 결과와 재개 지점','',
        '2026-09-18 04:51 KST 예약에서 재개. 기존 결과와 원본 프로토콜을 검토하고 개발자료 오경보 감사를 수행했다. 기존 P1/P2 및 실패한 P3 효과검정을 반복하지 않았다. 미사용자료에서 tracker 결과는 계산하지 않았다.','',
        '## 짧은 관측 신호의 전체 건강시점 비교','',
        f"건강한 시작 {alarm['healthy_landmarks']}시점 중 앞으로2초 안에 시작한 지속이탈 {alarm['sustained']}건({alarm['event_works']}작품군), 기본발생률 {alarm['base_rate']*100:.2f}%.",
        '| 직전2초 경고 규칙 | 이탈 포착 | 무이탈 오경보 | 오경보율 | 경고 중 실제이탈 |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('any_positive','B>0 한 프레임 이상'),('two_consecutive','B>0 연속 두 프레임')]:
        r=alarm[key];lines.append(f"| {label} | {r['tp']}/{r['tp']+r['fn']} | {r['fp']}/{r['fp']+r['tn']} | {r['false_alarm_rate']*100:.1f}% | {r['positive_predictive_value']*100:.2f}% |")
    lines+=['','이전의 “7건 모두 선행 B>0”은 같은 규칙을 무이탈에도 적용하면 판별력이 약하다. 모든 특징 프레임은 고정시점 전에 완료된다. 사건 당프레임 관측을 신호에 넣지 않았다.',
        '연속 두 프레임은 오경보를 줄이지만 사건2건을 놓친다. 두 규칙을 모두 공개하고 어느 쪽도 P3 성공 신호로 채택하지 않는다.','',
        '작품군 단위에 하나라도 경고가 있었는지:',
        '| 규칙 | 사건 작품군 경고 | 사건 없는 작품군 경고 |','|---|---:|---:|']
    for key,label in [('any_positive','한 프레임'),('two_consecutive','연속 두 프레임')]:
        r=family_summary[key];lines.append(f"| {label} | {r['alerted_event_families']}/{r['event_families']} | {r['alerted_no_event_families']}/{r['no_event_families']} |")
    lines+=['','## 보존자료의 적격성','',
        f"메타데이터·주석·MIDI 존재/길이만 검사한 결과 적격 미사용 작품군 {preflight['eligible_families']}개: 베토벤 {preflight['eligible_per_composer'].get('Beethoven',0)}, 쇼팽 {preflight['eligible_per_composer'].get('Chopin',0)}. 명시적 과거 manifest와 작품군 충돌 {len(preflight['explicit_manifest_family_collision'])}개.",
        '이는 명시적 결과 manifest에 대한 확인이다. 과거 모든 탐색·동일 작품집 스타일의 독립성을 보증하지 않는다. 보존자료의 chroma/cost/tracker/결과는 계산하지 않았다.','',
        '## 실제 변동을 반영한 표본 시나리오','',
        '계획상의 raw Brier 개선 0.004(기존 단순발생률 Brier≈0.079의 약5%)를 가정했다. 이 수치는 도메인 관점의 최소 의미효과로 확정된 값이 아니라 규모를 보여주는 예시다.',
        '| 개발모형 손실차 출처 | 작품별 SD | d=0.004/SD | 80% power 작품수 | 남은25작품 power |',
        '|---|---:|---:|---:|---:|']
    for source,item in zip(variance['source_summary'],variance['sensitivity'][:2]):
        lines.append(f"| {source['model']} | {source['sd']:.4f} | {item['standardized_d']:.3f} | {item['works_for80power']} | {item['power_with25']*100:.1f}% |")
    lines+=['','이는 독립·정규 작품손실차와 단측 α=.05 paired-t 근사다. 교차검증에서 관측한 손실차 SD는 불안정할 수 있으며 같은 작곡가 상관·희귀사건·모형훈련의 변동을 반영하지 않는다. 실제 필요 규모는 더 클 수 있다.',
        '기존 개발표본에서 현재위치 관측B 모형의 평균 Brier 개선은 음수였고, 복수후보 모형의 logloss는 단순발생률보다 나빴다. 가정한0.004 개선이 실제로 존재한다는 증거가 아니다.','',
        '## 진행 판정','',
        '**현재 P3 독립 검증 실행 중지.** 이미 고정된 현재위치 모형 gate가 실패했고, 짧은 경고는 높은 오경보율을 보였다. 적격25작품은 규모와 작곡가 다양성 모두 부족하다. 이를 지금 단회 테스트하면 모호한 음성결과를 양산하고 보존자료를 소모한다.',
        '다음 캠페인에 필요한 구체 조건: (1) 작품·작곡가가 넓게 분리된 새 연주와 정렬 GT, (2) 개발자료의 모든 건강시점에서 사건·무사건에 동일 경고를 적용한 실용적인 오경보율, (3) 도메인 관점에서 의미 있는 raw 손실개선과 그에 맞춘 표본계획, (4) 정답의 oracle 측정과 실제 온라인가용 측정을 구분한 단일 사전 프로토콜.',
        '새 자료가 도착하면 `scripts/p3_holdout_preflight.py`의 적격성·중복 검사 방식으로 먼저 이력을 확인한 뒤, 하나의 고정된 추적기·시계·결과·검정을 실행한다. 기존 보존25작품은 독립성 검토가 끝날 때까지 결과 미개봉으로 둔다.',
        '원래 자연 P3는 미검증. 인위적 소거 효과나 수학적 Bayes 항등식으로 증명 판정을 대체하지 않는다.','',
        '## 재현','',
        '- `python scripts/p3_short_signal.py` → `results/p3_onset_development/short_signal/`',
        '- `python scripts/p3_holdout_preflight.py` → `results/p3_onset_development/holdout_preflight/`',
        '- `python scripts/p3_sample_variance.py` → `results/p3_onset_development/sample_variance/`',
        '- `python -m pytest tests -q` → 74 passed.',
        '세 실행은 이미 완료됐다. 새 근거 없이 같은 개발자료의 threshold·horizon·후보수를 다시 맞추는 반복은 진행하지 않는다.']
    report='\n'.join(lines)+'\n';(ROOT/'P3_SCHEDULED_RUN_REPORT.md').write_text(report,encoding='utf-8');(root/'SCHEDULED_RUN_REPORT.md').write_text(report,encoding='utf-8')
    entry='''
## 2026-09-18 04:51 예약 재개 — 짧은 신호 오경보 및 독립검증 gate

전체 건강396시점,2초내 지속이탈7건. 선행B>0 한프레임 규칙: TP7/7,
FP323/389(오경보83.0%, PPV2.12%, 기본발생1.77%). 연속2프레임 규칙:
TP5/7, FP192/389(오경보49.4%, PPV2.54%). 사전 건강시점 동일규칙 비교.
미사용25작품(베토벤12/쇼팽13) 결과 미개봉. raw Brier .004 개선 가정과
개발작품별 손실차 SD로는80% 검증력에327~570작품 계획근사.
현재위치모형 기존 gate 미달·짧은 신호 오경보 과다·표본 부족으로
독립검증 실행을 보류. 자연P3 미검증 유지. 테스트74개 통과.
상세 [P3_SCHEDULED_RUN_REPORT.md](P3_SCHEDULED_RUN_REPORT.md).

'''
    for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
        path=ROOT/name;old=path.read_text(encoding='utf-8')
        if '## 2026-09-18 04:51 예약 재개' not in old:
            first,_,rest=old.partition('\n');path.write_text(first+'\n'+entry+rest,encoding='utf-8')
    snapshot=root/'scheduled_source_snapshot';snapshot.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'P3_SCHEDULED_RUN_REPORT.md',ROOT/'scripts/p3_short_signal.py',ROOT/'scripts/p3_holdout_preflight.py',ROOT/'scripts/p3_sample_variance.py',ROOT/'P3_SHORT_SIGNAL_PROTOCOL.md',ROOT/'tests/test_p3_short_signal.py']
    for f in files:shutil.copy2(f,snapshot/f.name)
    (root/'scheduled_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    print(report)

if __name__=='__main__':main()
