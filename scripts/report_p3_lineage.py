"""Persist mechanism findings separately from prospective P3 claims."""
import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from p3_temporal import read_csv
sys.stdout.reconfigure(encoding='utf-8')
out=ROOT/'results/p3_onset_development/lineage';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));rows=read_csv(out/'events.csv')
lines=['# P3 이탈 후보 계보 감사','',
    '기존 개발자료22사건의 사후 원인진단이다. 미사용 자료를 열지 않았고 새로운 자연P3 유의성 검정도 하지 않았다.',
    '사전 감사규칙: [P3_LINEAGE_PROTOCOL.md](P3_LINEAGE_PROTOCOL.md). 원본: `results/p3_onset_development/lineage/`.','',
    '## 핵심 결과','',
    '| 이탈 시작의 이동 유형 | 사건 수 | 이후 restart 제거로 방지 | 이후 restart 성분이 승자질량의 과반 |',
    '|---|---:|---:|---:|']
for kind,g in m['groups'].items():lines.append(f"| {kind} | {g['n']} | {g['restart_off_prevented']} | {g['new_majority']} |")
lines+=['','large_switch는 이탈 시작 순간 MAP 이동이 score .5초 초과. small_switch는 그 이하이지만 전이0~3frame 밖의 이동. local_crossing은 전이범위 안에서 오차기준을 넘음.',
    '이 유형은 사건 시작 순간의 기술적 구분이다. local_crossing 2건만으로 연속 누적 drift를 입증하지 않으며, large_switch도 그 이전의 점진적 불확실성 증가를 배제하지 않는다.',
    f"이탈 순간 승자 확률질량이 주로 기존 성분인 사건 {m['existing_lineage_majority']}건, 이후 restart 성분인 사건 {m['new_lineage_majority']}건. 전체 새성분 비율 중앙값 {m['median_new_share']:.8f}.",
    '기존 성분은 노출완료 시 이미 posterior에 있던 질량이다. 그 이전 restart에서 유래했을 가능성은 남아 있다.',
    'restart 제거 후 남은16건 중12건은 큰 후보전환이다. 따라서 restart를 끄는 것만으로 누적 drift만 남는 깨끗한 실험이 되지는 않는다.',
    '확률질량 계보에서 과반인8건과 restart제거로 방지한6건은 다른 양이다. 다른 후보가 대신 승리하거나 발생시점이 바뀔 수 있으므로 일대일 인과귀속으로 해석하지 않는다.','',
    '## 사건별 공개','',
    '| 작품 | 시점→이탈(초) | 유형 | MAP 이동(초) | 새성분 비율 | 정답영역 posterior | restart제거 후 이탈 |',
    '|---|---|---|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['composition']} | {float(r['landmark_sec']):g}→{float(r['onset_sec']):.1f} | {r['kind']} | {float(r['map_step_sec']):+.3f} | {float(r['new_share_at_MAP']):.6f} | {float(r['true_region_mass']):.6f} | {r['restart_off_failure']} |")
lines+=['','## 계산 검증과 해석 범위','',
    f"posterior=기존성분+새성분 복원 최대잔차 {m['max_posterior_residual']:.3g}. 이탈시점 사후승자/GT지점 odds 항등식 최대잔차 {m['max_odds_identity_residual']:.3g}. 원본 전체 재생과 사건 추적 MAP 일치. 전체66 tests passed.",
    '지점별 prior log odds, 당프레임 observation log BF, posterior log odds는 events.csv에, 시점별 MAP확률·정답영역질량·새성분비율은 traces.csv에 저장했다.',
    '이탈시점 승자는 미래결과를 보고 지정한 사후 대상이다. 이 odds와 trace는 예측변수로 사용하거나 미래를 미리 알 수 있었다는 근거로 제시하지 않는다.',
    '이 분석은 한 Bayes 추적기와 기존 개발자료에 한정된다. OLTW나 다른 자료의 실패분포까지 일반화하지 않는다.','',
    '## P3를 위한 방향 수정','',
    '현재 주된 실패를 단일 오차 누적식에 맞추기보다, 관측 약화가 기존 후보 간 확률우세의 전환에 앞서는지 검증하는 경로를 별도 가설로 둔다.',
    '원래 P3가 누적 drift를 특별히 주장한다면 후보전환 가설의 성공으로 자동 대체할 수 없다. 국소 오차축적과 후보전환을 모두 보고해야 한다.',
    '다음 개발 단계에서는 미래 승자를 쓰지 않고, 관측 이전에 정한 복수 후보영역의 질량·순위·관측증거를 추적한다. 기존의 단일 최대prior 오답만 선택한 지표가 나중에 부상하는 후보를 놓쳤는지 측정한다.',
    '원래22사건만 비교하면 선택 편향이 생기므로 건강한 노출완료371시점 전체에서 같은 절차를 적용한다. 이후 작품단위 교차검증으로 기준 오차모형 대비 신규 이탈/후보전환 예측력을 평가하고, 개발결과와 독립검증은 분리한다.',
    '미사용25후보 작품의 독립성·검증력·작곡가 편중을 해결하기 전 결과를 소비하지 않는다.']
report='\n'.join(lines)+'\n';(ROOT/'P3_LINEAGE_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
entry='''
## 2026-09-18 신규 이탈의 후보 계보 감사 완료

기존22사건: 이탈시점 큰MAP전환18, 작은전환2, local crossing2.
승자질량의 기존성분 과반14건, 노출완료 이후restart성분 과반8건.
restart제거로 방지6건은 모두 큰전환. 제거 후에도 큰전환12건이 남는다.
posterior 성분복원 최대잔차2.22e-16, odds항등식2.66e-15, 전체66 tests passed.
원본 경로 일치. 사후 기전감사이며 자연P3 증명/새 유의성 검정은 아님.
국소 누적drift와 후보전환을 구분하고 미래승자를 쓰지 않는 복수후보 관측전 추적을
다음 개발방향으로 기록. 미사용 자료 결과는 열지 않음.
상세 [P3_LINEAGE_REPORT.md](P3_LINEAGE_REPORT.md), 원본 `results/p3_onset_development/lineage`.

'''
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    if '## 2026-09-18 신규 이탈의 후보 계보 감사 완료' not in old:
        first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
files=[Path(__file__),ROOT/'P3_LINEAGE_REPORT.md']
for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
(out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print(report)
