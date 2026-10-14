import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.stdout.reconfigure(encoding='utf-8')
out=ROOT/'results/p3_onset_development/current_position';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
lines=['# P3 현재위치 관측기여 개발평가','',
    '기존371시점·35작품군·22신규이탈. 실행 전 [P3_CURRENT_POSITION_PROTOCOL.md](P3_CURRENT_POSITION_PROTOCOL.md)에 변수·모형·진행기준을 고정했다. 미사용자료는 실행하지 않았다.','',
    '## 결과','',
    '| 모형 | Brier ↓ | Log loss ↓ |','|---|---:|---:|']
for name,r in m['models'].items():lines.append(f"| {name} | {r['brier']:.6f} | {r['logloss']:.6f} |")
lines+=['','C=훈련작품별 가중발생률, S=현재오차·추세·오답posterior질량·entropy, B=S+평균관측 log BF, D=S+평균오답질량 변화.',
    '관측 B>0은 현재오답영역을 더 지지한다는 뜻이며 D>0은 새관측으로 오답질량이 증가했다는 뜻이다.','',
    '| 비교 | Brier 상대개선 | Log loss 상대개선 | Brier 개선작품 |','|---|---:|---:|---:|']
for name,r in m['comparisons'].items():lines.append(f"| {name} | {r['brier_improvement_percent']:+.2f}% | {r['logloss_improvement_percent']:+.2f}% | {r['improved_works']}/35 |")
lines+=['','양수=손실감소. 주모형 B의 독립검증 진입 gate: '+('통과' if m['independent_validation_gate'] else '**미달**')+'.',
    'gate는 B가 S와 C 둘 다에 대해 Brier/logloss 각각5% 이상 개선하고 S 대비 개선작품이 과반이라는 사전 운영기준이다. 통계적유의성 기준이나 자연P3의 진릿값 판정은 아니다.',
    'B는 logloss만 개선되고 Brier는 악화했다. D는 두 평가 모두 상태기준보다 악화했다. 지표/모형을 사후 선택해 성공으로 보고하지 않는다.','',
    '## 관측은 잘못된 위치를 지지했는가','',
    '| 결과 | n | 평균B 중앙값 | 평균D 중앙값 | 평균B>0 |','|---|---:|---:|---:|---:|']
for label in ['failure','no_failure']:
    r=m[label];lines.append(f"| {label} | {r['n']} | {r['median_B']:+.6f} | {r['median_D']:+.8f} | {r['positive_mean_B']} |")
lines+=['','이탈군의 B는 덜 음수이므로 정답을 지지하는 관측우세가 상대적으로 약한 경향이다. 하지만22건 중21건에서 평균B<0, 즉 측정구간 관측은 여전히 정답영역을 지지했다.',
    'D는 prior의 오답질량 수준에도 좌우된다. 이탈군에서 더 큰 음의 D가 나왔다고 더 좋은 추적상태였다는 뜻은 아니다. 정보척도 B와 확률변화 D가 다른 개념임을 보여준다.',
    '이 집단차이는 개발자료 기술통계다. 현재상태를 넘는 미래예측력이 확보됐다는 근거로 바꾸지 않는다.','',
    '## 검증 및 한계','',
    f"동일 현재영역에 대한 prior/posterior odds 항등식 최대잔차 {m['max_identity_residual']:.3g}. 전체72 tests passed. 40개 원본경로 일치, 무정보·부호·낮은prior질량 대조 통과.",
    '각시점 correct영역은 현재GT±.5score초. 프레임마다 영역이 움직이므로 평균B는 고정가설의 누적 Bayes factor가 아니다.',
    '35작품군 leave-one-out 및 train-only 표준화. 반복적으로 탐색한 개발자료이므로 확증 p-value/CI를 제시하지 않는다.',
    '현재오차/상태를 통제하는 것은 추가예측력 질문에 적합하지만 관측효과가 상태를 통해 전달되는 경로도 통제한다. 이 미달만으로 관측의 전체 인과효과가 없다고 주장하지 않는다.',
    '노출종료까지 건강했던371시점에 조건부이며 노출중25건 임계값초과는 별도다. GT보간·전체MIDI길이 정규화의 oracle전처리 및 Bayes단일추적기 한계가 있다.','',
    '## 이번 단계의 결정','',
    '사전에 고정한 gate 미달이므로 이 모형을 남은 미사용자료에 돌리지 않았다. 초기후보와 현재위치의 혼동은 수정했지만, 자연P3의 독립예측력이라는 핵심은 여전히 미검증이다.',
    '동일한22개 사건으로 척도·시간창·모형을 계속 바꾸는 방식은 개발자료에 맞출 위험이 커졌다. 현재 결과를 하나의 실패/혼합근거 묶음으로 고정한다.',
    '다음 독립캠페인은 관측2초 종료 뒤의 이탈예측이라는 제한된 질문과, 약화도중 시작하는 이탈을 포함한 전체 시간과정 질문을 구분해 설계해야 한다. 더 긴 독립연주/작품과 충분한 사건수를 확보하기 전 기존자료 수치로 P3 증명을 선언하지 않는다.']
report='\n'.join(lines)+'\n';(ROOT/'P3_CURRENT_POSITION_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
entry='''
## 2026-09-18 현재위치 관측기여 개발평가 — 독립검증 gate 미달

371시점/35작품군/22이탈. 현재GT±.5초에 대해 prior/posterior를 동일영역 비교.
주모형B vs 상태S: Brier상대개선−0.24%, logloss+4.06%; vs 발생률C: −4.74%, +3.29%.
보조D vs S: Brier−2.99%, logloss−3.82%. 양평가지표/양기준 각각5% 개선 gate 미달.
이탈군 관측BF 중앙값−.607, 무이탈−1.694. 이탈22중21건도 평균관측은 정답지지.
약한 관측우세 경향과 추가예측력은 구분. 자연P3 여전히 미검증.
전체72 tests passed, 항등식잔차1.41e-14, 원본경로일치. 미사용자료 실행 안 함.
상세 [P3_CURRENT_POSITION_REPORT.md](P3_CURRENT_POSITION_REPORT.md),
원본 `results/p3_onset_development/current_position`.

'''
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    if '## 2026-09-18 현재위치 관측기여 개발평가' not in old:
        first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
files=[Path(__file__),ROOT/'P3_CURRENT_POSITION_REPORT.md']
for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
(out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print(report)
