import json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from p3_temporal import read_csv
sys.stdout.reconfigure(encoding='utf-8');out=ROOT/'results/p3_onset_development/timing_sample';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));rows=read_csv(out/'timing.csv')
lines=['# P3 측정중 이탈 감사와 독립표본 계획','',
    '실행 전 [P3_TIMING_SAMPLE_PROTOCOL.md](P3_TIMING_SAMPLE_PROTOCOL.md). 개발자료만 분석했고 미사용자료 결과는 열지 않았다.','',
    '## 25개 제외시점은 무엇이었나','',
    f"측정2초 안에 오차 .5초를 한번이라도 넘은25시점 중, 그 구간에서 시작한5frame 지속이탈은 {m['sustained_windows']}건({m['sustained_works']}작품군), 나머지는 {m['transient_only']}시점이다.",
    '18시점이라는 수는 노출구간에서 시작한 지속이탈이 없었다는 뜻이다. 그 뒤에도 이탈이 전혀 없었다는 뜻으로 읽으면 안 된다.',
    f"7건의 시작은 측정시작 후 중앙값 {m['median_onset_delay_sec']:.1f}초. 지속확인이 노출종료 뒤에 완료된 사건은 {m['confirmation_after_exposure']}건으로, 확인지연과 실제시작을 구분했다.",
    f"7건 중 선행구간에 B>0 프레임은 {m['positive_B_before']}건, 직전2초 평균B>0은 {m['positive_pre2_mean_B']}건, 이탈당시 B>0은 {m['positive_onset_B']}건.",
    'B>0은 해당 프레임 관측이 오답영역을 더 지지함을 뜻한다. 평균B<0이어도 짧은 양의신호가 있을 수 있다. 그러나 이탈당프레임 관측은 이탈보다 앞선 정보가 아니므로 선행성 근거에 넣지 않는다.',
    '선행B 탐색범위는 고정시작 전2초부터 실제onset 직전까지라 사건별 길이가 다르다. 미래사건에 정렬한 진단이며, 무이탈 구간에서도 B>0이 얼마나 흔한지 비교하지 않은 민감도 수치다. 예측성공이나 인과성으로 해석하지 않는다.','',
    '| 작품 | 시작→이탈 | 지속확인 | 첫 선행B>0의 lead | 직전2초 평균B | 이탈프레임B |','|---|---|---:|---:|---:|---:|']
for r in rows:
    if r['sustained']!='True':continue
    lead=float(r['first_positive_lead_sec']) if r['first_positive_lead_sec'] else 0
    lines.append(f"| {r['composition']} | {float(r['time_sec']):g}→{float(r['sustained_onset_sec']):.1f} | {float(r['confirmed_sec']):.1f} | {lead:.1f}s | {float(r['pre2_mean_B']):+.4f} | {float(r['onset_B']):+.4f} |")
lines+=['','## 필요한 작품수: 가정별 계획값','',
    '작품별 평균 예측손실차의 단측 paired t 계획근사(alpha .05)를 사용했다. d=손실차 평균/작품간 손실차 표준편차. **d는 Brier의 상대개선율이 아니다.**',
    '독립·정규 작품차이 가정에서 noncentral-t로 계산했다. 정의와 구현은 [statsmodels TTestPower 문서](https://www.statsmodels.org/dev/generated/statsmodels.stats.power.TTestPower.power.html) 및 [공식 구현](https://www.statsmodels.org/stable/_modules/statsmodels/stats/power.html)을 참고했다.','',
    '| 가정 d | 80% power 작품수 | 90% power 작품수 | 25작품 power |','|---:|---:|---:|---:|']
for d in [.2,.35,.5]:
    pp=[r for r in m['sample_scenarios'] if r['standardized_effect']==d]
    lines.append(f"| {d:.2f} | {pp[0]['works']} | {pp[1]['works']} | {pp[0]['power_with25']*100:.1f}% |")
lines+=['','이 표는 가능한 평균차를 검출하는 계획근사다. 현재 모형이 그만큼 개선된다는 가정이나 실측 검증력 추정이 아니다. 실제 독립검증에서는 최소한 의미 있는 raw 손실개선과 예상분산을 먼저 정해야 한다.',
    '훈련모형을 고정하고 독립작품에서 평가하는 경우를 상정한다. 개발자료 교차검증의 상관된 fold차이를 독립표본으로 사용해 power가 확보됐다고 주장하지 않는다.',
    '예를 들어 d=.35, 80% power는52작품이다. 이는 작업규모를 가늠하는 시나리오이지 확정 목표가 아니다. 두 지표 동시성공·다중비교·비정규손실·같은작곡가 상관은 실제계획에서 추가 반영해야 한다.',
    '작품당 독립이탈이 .25/.5/1건이면52작품에서 기대13/26/52건이다. 이는 단순기대수이며 충분한 사건수나 검증력 보장이 아니다. 상세6개 시나리오는 sample_scenarios.csv에 저장했다.',
    '남은25후보작품은 두작곡가에 편중되고 완전미사용 이력도 확정되지 않았다. 작품수만으로도 이 가정들에서80% 기준에 못 미치며, 긴 연주로 window를 늘려도 독립작품수가 늘지는 않는다.','',
    '## 다음 독립캠페인 설계안','',
    '1. 이탈종료 후가 아니라, 매 예측시점 이전에 완료된 관측만 사용한다. 현재건강 상태를 확인한 시점마다 미래1초·5초 중 주시계 하나를 개발자료에서 정하고 고정한다. 사건은 .5초 지속으로 확인하되 onset시각과 확인시각을 따로 보존한다.',
    '2. 급격한후보전환과 국소편차증가를 별도 보조결과로 보고하고, 주결과는 모든 신규지속이탈로 둔다. 회복실패는 섞지 않는다.',
    '3. 짧은 관측이상 신호를 쓸지 여부는 건강한 모든시점의 오경보와 함께 개발한다. 이번7사건에만 맞춘 cutoff/시간창 선택으로 확증시험을 대체하지 않는다.',
    '4. 학습·설계용자료와 독립검증작품을 분리하고, raw 최소효과·분산·작품간상관·이탈빈도 시나리오를 포함한 시뮬레이션으로 최종표본수를 정한다. 현재27/52/156은 이 단계의 계획근사다.',
    '5. 개발지표를 동결하고 새자료를 한번 평가한다. 새자료의 중간효과로 중단·증원·모형교체를 하지 않는다. 품질/이력 탈락기준은 결과 전에 고정한다.','',
    '## 판정','',
    '이전설계가 빠른지속이탈7건을 별도분류했던 사실은 확인했다. 이를 포함하면 P3가 통과한다는 결론은 나오지 않았다. 자연P3는 여전히 미검증이다.',
    '현재 부족한 것은 단순 데이터량만이 아니다. 짧은 신호의 오경보율, 고정된 최소효과, 완전히 독립된 작품확보가 함께 필요하다.',
    '원본저장경로 일치 확인. power(d=0)=alpha 및 N/d에 대한 단조성 확인. 기존72개 테스트에 영향을 주는 추적기 변경은 없다.']
report='\n'.join(lines)+'\n';(ROOT/'P3_TIMING_SAMPLE_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
entry='''
## 2026-09-18 측정중 이탈 시간감사·독립표본 계획

노출중 임계값초과25시점 중 노출구간에서 시작한 지속이탈7건/5작품군,
나머지18시점은 해당구간 지속이탈시작 없음. 시작지연 중앙값1.4초, 확인이 노출뒤1건.
7건 모두 선행B>0 프레임, 직전2초 평균B>0은2건. 사후정렬·무이탈오경보 미평가라
예측성공 근거로 쓰지 않음. 자연P3 미검증 유지.
독립작품 손실차 정규/단측alpha.05 계획근사: d=.2/.35/.5일 때80% power에
156/52/27작품,90%에216/72/36작품. raw손실개선과 d는 다르며 확정표본수 아님.
미사용25후보작품 결과 보존. 시간순서/오경보/최소효과를 고정한 독립캠페인 필요.
상세 [P3_TIMING_SAMPLE_REPORT.md](P3_TIMING_SAMPLE_REPORT.md),
원본 `results/p3_onset_development/timing_sample`.

'''
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    if '## 2026-09-18 측정중 이탈 시간감사·독립표본 계획' not in old:
        first,_,rest=old.partition('\n');p.write_text(first+'\n'+entry+rest,encoding='utf-8')
files=[Path(__file__),ROOT/'P3_TIMING_SAMPLE_REPORT.md']
for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
(out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print(report)
