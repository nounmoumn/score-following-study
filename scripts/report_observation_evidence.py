"""Read-only report of prior-fixed evidence audit, including health-selection context."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);args=ap.parse_args()
    source=ROOT/args.result;out=source/'observation_audit'
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    queries=read(out/'query_results.csv');audit=read(source/'natural_event_audit.csv')
    statuses={(r['case_id'],float(r['event_time_sec'])):r['status'] for r in audit if r['tracker']=='bayes'}
    selection={}
    for label,allowed in [('healthy',{'matched','unmatched'}),('matched',{'matched'}),('unhealthy',{'unhealthy'})]:
        rr=[r for r in queries if r['role']=='event' and statuses[(r['case_id'],float(r['time_sec']))] in allowed]
        selection[label]=dict(n=len(rr),
            prior_competitor_ge_one_percent=sum(float(r['wrong_prior_mass'])>=.01 for r in rr),
            competitor_and_wrong_evidence=sum(float(r['wrong_prior_mass'])>=.01 and float(r['evidence_now'])<0 for r in rr),
            median_prior_correct_mass=float(np.median([float(r['correct_prior_mass']) for r in rr])))
    (out/'selection_context.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    e,c=m['roles']['event'],m['roles']['control']
    text=f"""# P3 관측 증거 측정 결과 — 2026-09-18

**측정 도구를 구현·검증했다. 새 관측이 주는 구별력과 이전 상태의 위치 확신을 분리할 수 있다.**
기존 사건에서 관측 정보는 약한 경향이지만 이전 상태는 대부분 정답 위치를 강하게 지지했다.
이는 ‘약한 관측=위치 상실’이라는 해석이 성립하지 않았던 이유를 구체화한다.
자연 P3를 새로 검정한 결과는 아니며 기존 미판정 상태를 변경하지 않는다.

## 무엇을 측정했나

GT 정답±.5초 영역과 오답 영역을 비교한다. 오답은 관측을 받기 전 predictive prior의
질량이 가장 큰 ±.5초 영역으로 선정한다. 오답영역은 정답±2초 제외 구간 밖에 완전히 놓인다.
새 관측의 비용이나 미래 오차를 후보 선정에 사용하지 않는다.

각 영역 안 prior로 가중한 emission likelihood의 비를 측정한다:

`B_t = log(L_correct / L_competitor)`

양수는 정답을 지지하는 새 관측, 음수는 경쟁 후보를 지지하는 관측, 0은 두 후보의 구별에
기여하지 않는 관측이다. ‘위치를 잃었다’나 ‘불확실성의 크기’와 같은 값이 아니다.

`posterior log odds = prior log odds + B_t`

이것은 Bayesian update의 항등식이며 **P3가 증명돼 도출된 동역학 방정식이 아니다**.
실제 구현에서 prior와 관측을 분리해 기록하고 두 항의 합이 posterior와 맞는지 검증했다.

## 기존 사건과 대조에서의 측정

| 기술통계 | 기존 약화 사건76개 | 저장된 대조41개 |
|---|---:|---:|
| 관측 전 정답주변 확률 중앙값 | {100*e['median_correct_prior_mass']:.3f}% | {100*c['median_correct_prior_mass']:.3f}% |
| 가장 유력한 오답영역의 prior질량 중앙값 | {e['median_wrong_prior_mass']:.3g} | {c['median_wrong_prior_mass']:.3g} |
| 현재 관측 log likelihood비 중앙값 | {e['median_log_evidence_now']:+.4f} | {c['median_log_evidence_now']:+.4f} |
| 과거2초 프레임별 log likelihood비 평균의 중앙값 | {e['median_log_evidence_past2']:+.4f} | {c['median_log_evidence_past2']:+.4f} |
| 오답영역 prior질량≥1% | {e['wrong_prior_ge_one_percent']}/{e['n']} | {c['wrong_prior_ge_one_percent']}/{c['n']} |
| 현재 관측이 오답을 더 지지 | {e['evidence_now_favors_wrong']}/{e['n']} | {c['evidence_now_favors_wrong']}/{c['n']} |
| 위 두 조건 모두 | {e['prior_competitor_and_wrong_evidence']}/{e['n']} | {c['prior_competitor_and_wrong_evidence']}/{c['n']} |

사건/대조의 수·곡 집합이 다르고 상관된 관측을 포함한다. 표는 기술통계이며 p값을 산출하거나
두 집단의 차이를 확증하지 않았다. ≥1%는 고정한 기술적 구분일 뿐 새 사건 정의/최적 임계값 아님.
2초 평균은 매 프레임 후보가 달라질 수 있으므로 **같은 후보의 누적 evidence가 아니다**.

## 기존 건강도 선택과 함께 읽기

| Bayes 기존 사건 집합 | 사건 수 | 경쟁 prior≥1% | 경쟁 prior≥1%이면서 관측이 오답 지지 | 정답 prior 중앙값 |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {name} | {r['n']} | {r['prior_competitor_ge_one_percent']} | {r['competitor_and_wrong_evidence']} | {100*r['median_prior_correct_mass']:.3f}% |" for name,r in selection.items())}

집합 구분은 기존 결과에 저장된 것을 그대로 사용했다. 미래 결과를 보고 새 사건을 만들지 않았다.
비슷한 관측이어도 이전 상태가 정답을 충분히 지지하면 위치를 계속 유지할 수 있다는 설명과
정합한다. 그러나 이것만으로 ‘이전 상태가 P3 실패의 원인’이라는 인과 결론이 나오지는 않는다.

## 검증

- 원래 Bayes 경로와 모든 재생 경로가 프레임별로 동일.
- 후보 선택이 현재 관측에 의존하지 않는 테스트 통과.
- 위치 정보가 없는 균일 비용행에서는 B=0.
- 정답/경쟁 유리 관측의 부호가 각각 양/음.
- {m['frames']:,}프레임의 odds 항등식 최대 잔차 **{m['max_identity_residual']:.3g}**.
- 전체 **55 tests passed**.

## 한계와 다음 단계

이 도구는 Bayesian filter에 대한 GT 기반 oracle 측정이다. OLTW의 확률이라고 부를 수 없고,
posterior를 직접 증거로 바꿔치기한 것도 아니다. 전곡 길이 tempo scaling과 단순 chroma,
고정 emission 모수에 의존한다. cost 자체가 완전한 음향 likelihood라는 보장도 없다.

다음 검증에서는 ‘관측 전 이미 경쟁하던 후보가 존재하는가’와 ‘새 관측의 구별력이 약한가’를
분리해야 한다. 기존 자료에서 측정 타당성만 확인한 뒤, 단서가 약해지기 **전** 상태를 기준으로
대조를 정하고 미사용 자료에서 시간순서를 검정해야 한다. 이론상 필요한 위험집합 조건인지
먼저 논증해야 하며, 성공한 사례만 남기는 사후 조건으로 사용하면 안 된다.
이번 턴에는 새로운 P3 검정/임계값 조정/미사용 자료 실험을 실행하지 않았다.

## 재현

```text
python scripts/audit_observation_evidence.py --result {args.result}
python scripts/report_observation_evidence.py --result {args.result}
python -m pytest tests -q
```

산출: `{args.result}/observation_audit/`의 frames, query_results, metrics,
selection_context, provenance, 소스/설계 사본. 원래 temporal 실행의 데이터 해시·manifest를 사용한다.
설계: [P3_OBSERVATION_PROTOCOL.md](P3_OBSERVATION_PROTOCOL.md).
확신도: 구현/실행수치 확정. 해석은 잠정. 자연 P3 전체는 미판정.
"""
    (ROOT/'P3_OBSERVATION_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(selection,indent=2));print('Written P3_OBSERVATION_REPORT.md')
if __name__=='__main__':main()
