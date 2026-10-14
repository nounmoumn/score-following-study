import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
out=ROOT/'results/p3_onset_development/multicandidate';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
lines=['# P3 복수 후보 개발 검증','',
    '관측 전에 고정한 오답5후보를 기존 건강한371시점 전체에서 추적했다. 신규이탈22건. 미래 승자를 후보선정에 사용하지 않았다.',
    '실행 전 [P3_MULTICANDIDATE_PROTOCOL.md](P3_MULTICANDIDATE_PROTOCOL.md). 원본 `results/p3_onset_development/multicandidate`.',
    f"동일 작품 악장·반복버전을 묶은 {m['works']}작품군 leave-one-out. 개발자료를 반복해서 탐색한 결과이므로 독립확증 검증으로 부르지 않는다.",'',
    '## 작품균등 예측손실','',
    '| 모형 | Brier ↓ | Log loss ↓ |','|---|---:|---:|']
for name,r in m['models'].items():lines.append(f"| {name} | {r['brier']:.6f} | {r['logloss']:.6f} |")
lines+=['','constant=훈련작품의 가중 발생률, B=과거·노출완료 오차 및 prior상태, S=B+1순위 후보 W, M=S+5후보 최대W 및 최대노출완료 odds.',
    '각 fold의 표준화·적합·상수발생률은 훈련작품만 사용했다. alpha1 고정 logistic ridge이며 모형/임계값 탐색을 하지 않았다.','',
    '| 비교 | Brier 개선 | 상대개선 | 개선 작품군 |','|---|---:|---:|---:|']
for name,r in m['comparisons'].items():lines.append(f"| {name} | {r['brier_improvement']:+.6f} | {r['relative_percent']:+.2f}% | {r['improved_works']}/{m['works']} |")
lines+=['','양수 개선은 예측손실 감소. 개발평가라 p-value나 확증적 신뢰구간을 계산하지 않았다.','',
    '## 초기1순위 후보가 끝까지 대표적인가','',
    '| 분류 | n | 노출완료 최고후보가 초기1순위 아님 | 진단종료 최고후보가 초기1순위 아님 | 노출완료 정답가설 추월 | 진단종료 추월 |',
    '|---|---:|---:|---:|---:|---:|']
for name,r in m['diagnostics'].items():lines.append(f"| {name} | {r['n']} | {r['end_best_not_top1']} | {r['later_best_not_top1']} | {r['end_overtake']} | {r['later_overtake']} |")
lines+=['','진단종료는 이탈사건에서는 이탈시작, 무이탈에서는5초 추적종료다. 서로 다른 종료규칙이므로 해당 비율 차이를 동등한 시계의 예측성능으로 해석하지 않는다. 미래진단 변수는 예측모형에 들어가지 않는다.',
    '가설점수=초기영역 prior질량×그 영역에서 시작한 전체경로의 관측가능도. restart 이후 경로가 합쳐질 수 있어 점수추월은 최종위치영역 posterior 추월이나 실제 MAP전환과 동일하지 않다.',
    '또한5개밖의 후보에서 실제 이탈이 발생할 수 있다. 후보점수 진단만으로 실제승자 포착률을 주장하지 않는다.','',
    '## 판정과 한계','',
    '이번 검증은 단일후보 지표의 대표성 및 추가 예측력을 개발자료에서 점검한다. 원래 누적drift P3와 후보전환 예측가설은 계속 구분한다.',
    '노출2초 동안 건강했던시점에 조건부이므로 노출중 탈락25시점이나 전체 인과효과로 일반화하지 않는다. oracle GT·전체연주길이 정규화·한 Bayes tracker의 한계도 유지한다.',
    '상수모형을 포함한 모든 모형 결과를 공개하며 복수후보가 나쁘면 그대로 기록한다. 남은 미사용작품 자료는 실행하지 않았다.',
    '검증: 전체69개 테스트 통과. 각 시점1순위 W는 이전계산과 일치하며 전체재생 MAP는 기존저장 경로와 일치. 후보 비중복·forward 가능도·logistic 기준값 테스트 통과.']
improvement=100*(m['models']['constant']['brier']-m['models']['M']['brier'])/m['models']['constant']['brier']
lines+=['','## 이번 결과의 구체적 해석','',
    f"복수후보 M의 Brier는 기준B 대비 {m['comparisons']['M_vs_B']['relative_percent']:.2f}%, 단일후보S 대비 {m['comparisons']['M_vs_S']['relative_percent']:.2f}% 감소. 단순 발생률 constant 대비 감소는 {improvement:.2f}%다.",
    '반면 M의 logloss는 S 및 constant보다 높다. 두 지표가 일관되게 개선된 결과가 아니므로 예측력이 확실히 확보되었다고 단정하지 않는다.',
    '이탈22건 중 관측완료 시점에서 최고 증거점수가 초기1순위가 아니었던 경우는5건. 무이탈349시점에서도157건이었다. 후보순위 교체 자체를 위험징후라고 단정할 수 없다.',
    '이탈시점까지 고정 초기영역 가설의 정답추월은3/22건. 실제 MAP전환 대부분과 이 지표의 가설추월이 일치하지 않는다는 측정상의 간극이 남는다.',
    '다음 진단은 초기영역 가설과 최종위치 posterior를 구분해 연결해야 한다. 후보를 더 늘리거나 미래승자를 넣어 결과를 개선하는 대신 실제승자에 대한 사전후보 포괄성과 restart 이후 질량혼합을 분리해 감사한다.']
report='\n'.join(lines)+'\n';(ROOT/'P3_MULTICANDIDATE_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
entry=f"\n## 2026-09-18 복수후보 개발 검증 완료\n\n건강한371시점/22이탈/{m['works']}작품군 leave-one-out. 관측전 오답5후보 고정, train-only 표준화 logistic alpha1.\n"
for name,r in m['comparisons'].items():entry+=f"- {name}: Brier상대개선 {r['relative_percent']:+.2f}%, 개선작품 {r['improved_works']}/{m['works']}.\n"
entry+='\n전체69 tests passed. 독립확증시험/자연P3 증명 아님. 미래진단순위는 예측에 사용하지 않음. 미사용자료 결과 미개봉.\n상세 [P3_MULTICANDIDATE_REPORT.md](P3_MULTICANDIDATE_REPORT.md), 원본 `results/p3_onset_development/multicandidate`.\n\n'
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    if '## 2026-09-18 복수후보 개발 검증 완료' not in old:
        first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
files=[Path(__file__),ROOT/'P3_MULTICANDIDATE_REPORT.md']
for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
(out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print(report)
