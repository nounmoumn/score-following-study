import json,sys,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from p3_temporal import read_csv
sys.stdout.reconfigure(encoding='utf-8')
out=ROOT/'results/p3_onset_development/coverage';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));rows=[r for r in read_csv(out/'events.csv') if r['phase']=='onset']
labels={'correct_initial':'정답근처 출발','selected_wrong':'선택한 오답후보','unselected':'나머지 위치','new_restart':'이후 restart'}
lines=['# P3 초기 후보와 최종 위치의 불일치 감사','',
    '기존 개발자료22이탈의 사후 기전감사. 미래승자를 예측변수로 사용하지 않았으며 미사용자료도 실행하지 않았다.',
    '실행 전 [P3_COVERAGE_PROTOCOL.md](P3_COVERAGE_PROTOCOL.md). 원본 `results/p3_onset_development/coverage`.','',
    '## 실제 이탈위치 확률의 출처','',
    '| 확률질량 출처 | 승자위치에서 과반인 사건 |','|---|---:|',
    f"| 초기 정답근처 영역 | {m['correct_initial_majority']} |",
    f"| 초기 선택한 오답5영역 | {m['selected_wrong_majority']} |",
    f"| 초기 나머지 영역 | {m['unselected_majority']} |",
    f"| 관측 시작 이후 restart | {m['new_restart_majority']} |",'',
    '네 성분은 서로 중복 없이 전체 posterior를 복원한다. 이번22건은 모두 한 성분이 과반이었다. 과반은 기술요약이며 다른 성분의 기여가0이라는 뜻은 아니다.',
    '새restart 기준점은 관측시작이다. 이전 계보감사의 노출완료 이후 기준과 다르므로 같은 수가 나와도 분해조건이 동일한 것은 아니다.',
    '정답근처 출발10건은 후보가 출발한 위치를 말한다. 이후 그 후보의 경로가 실제정답에 남아 있다는 뜻이 아니다. 이 범주의 존재를 누적drift P3가 입증됐다는 뜻으로 해석하지 않는다.',
    '## 왜 가설점수 추월은3건인데 위치이탈은22건인가','',
    '기존 likelihood 점수는 초기영역 가설의 상대적 가능도다. 초기 정답영역에서 출발한 경로가 잘못된 위치로 이동해도 여전히 정답출발 가설에 속한다. 이때 잘못된 현재위치가 MAP가 되어도 초기오답 가설이 추월할 필요가 없다.',
    'restart는 서로 다른 초기영역 가설의 경로를 같은 위치로 섞을 수 있다. 초기 출처별 posterior와 가장 최근 restart 이후 출처별 posterior는 서로 다른 분해다.',
    f"선택한오답 출처가 이탈위치 질량의 과반인4건 중 {m['selected_wrong_majority_without_origin_overtake']}건도 초기오답 가설의 전체질량 추월 없이 MAP전환이 발생했다. 전체 가설질량과 한 위치에서의 최대확률 역시 다르다.",
    '따라서 단순히 오답후보 개수를 늘리는 것만으로 이 불일치가 해소된다는 근거는 없다. 나머지영역 출처가 과반인 사건은0건이지만, 포괄성이 완벽하거나 미래자료에서도 충분하다는 증명은 아니다.','',
    '## 사건별 값','',
    '| 작품 | 이탈시각 | 지배적 출처 | 비중 | 정답출발 | 오답5개 | 나머지 | 새restart |','|---|---:|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['composition']} | {int(r['frame'])/10:.1f} | {labels[r['dominant']]} | {float(r['dominant_share']):.5f} | {float(r['correct_initial_winner_share']):.5f} | {float(r['selected_wrong_winner_share']):.5f} | {float(r['unselected_winner_share']):.5f} | {float(r['new_restart_winner_share']):.5f} |")
lines+=['','## 검증','',
    f"전체 posterior 복원 최대잔차 {m['max_posterior_residual']:.3g}. 초기영역 분해 odds와 이전 conditional likelihood 점수의 최대차 {m['max_origin_odds_residual']:.3g}. 실제 경로도 원본과 일치. 전체70 tests passed.",
    '미래승자 위치에서 계산한 출처비율은 사후설명이다. 그 값으로 신규이탈을 미리 예측했다고 주장하지 않는다.','',
    '## 다음 검증의 초점','',
    '관측 완료 시점의 **현재 위치분포**가 실제정답±.5초 안팎에 얼마나 놓이는지 측정하고, 그 바깥 질량을 transition만 적용했을 때와 observation까지 적용했을 때로 나눠 기록한다.',
    '이렇게 하면 관측이 오차 증가를 억제하는지, 오히려 잘못된 위치에 질량을 싣는지 같은 시점에서 분리할 수 있다. 미래GT를 미리 넣지 않는 과거/현재 oracle 진단이며 실제배치 지표와 구분한다.',
    '개발단계에서는 건강한371시점 전체에 동일한 측정을 적용하고 이후5초 신규이탈의 작품단위 예측력을 평가한다. 현재 오차/불확실성만으로 설명되는지도 기준모형에 포함한다.',
    '현재위치 기준의 관측 기여와 미래 발생의 시간적 선행성이 독립자료에서도 확인되어야 P3의 근거가 된다. 계보항등식이나 이번사후설명만으로는 증명되지 않는다.']
report='\n'.join(lines)+'\n';(ROOT/'P3_COVERAGE_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
entry='''
## 2026-09-18 초기후보 포괄성·현재위치 불일치 감사

개발22이탈의 승자위치 질량출처 과반: 초기정답근처10, 선택오답5영역4,
나머지초기영역0, 관측시작후restart8. 초기출처와 현재위치는 다르므로
초기정답가설이 우세해도 그 경로가 현재정답에서 벗어날 수 있음.
전체 posterior 복원잔차7.77e-16, 초기출처 odds/이전가능도 일치7.11e-14.
전체70 tests passed. 사후기전 설명이며 자연P3 증명/새예측평가 아님.
다음은 현재위치 오차영역의 prior→posterior 질량변화로 관측기여를 측정.
상세 [P3_COVERAGE_REPORT.md](P3_COVERAGE_REPORT.md), 원본 `results/p3_onset_development/coverage`.

'''
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    if '## 2026-09-18 초기후보 포괄성·현재위치 불일치 감사' not in old:
        first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
for f in [Path(__file__),ROOT/'P3_COVERAGE_REPORT.md']:shutil.copy2(f,out/'source_snapshot'/f.name)
(out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'P3_COVERAGE_REPORT.md']},indent=2),encoding='utf-8')
print(report)
