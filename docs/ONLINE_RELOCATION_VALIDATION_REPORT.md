# 온라인 relocation 검증 실험 보고서

작성일: 2026-09-24  
사전 계약: `ONLINE_RELOCATION_VALIDATION_PROTOCOL.md`

## 결론

**큰 위치 jump 뒤 2초 동안 후속 상태와 연주 관측을 확인하는 validator는 실용적으로 유망했다.** work-equal decision regret을 0.2484에서 0.1164로 절반 이상 줄였고, bootstrap 95%도 양의 개선을 보였다. 7작품 중 개선 3, 동률 4, 악화 0이었다.

그러나 사전 gate는 7작품 중 최소 5작품의 엄격한 개선을 요구했다. 네 작품은 비교기의 regret이 이미 0이라 개선할 수 없었고 동률이었지만, 규칙상 성공으로 셀 수 없다. 따라서 공식 판정은 **유망하지만 development gate 미통과**, lockbox 미개봉이다.

## 쉽게 말하면

추적기가 갑자기 다른 곳으로 점프하면 두 선택이 있다.

- 바로 새 위치를 믿는다.
- 2초 동안 지켜본 뒤, 새 위치가 안정적이고 다른 추적기와 연주 음표가 지지하면 받아들인다.

첫 번째 방식은 거의 항상 jump를 받아들였다. 두 번째 방식은 문제가 있던 세 작품에서 잘못된 jump 일부를 걸러냈고, 원래 문제가 없던 네 작품에서는 결과를 망치지 않았다.

따라서 이전의 static ambiguity 방향보다 실제 행동과 훨씬 직접적으로 연결된다. 다만 지금 자료만으로 lockbox를 열 만큼 사전 조건을 충족한 것은 아니다.

## 설계

- `arzt`와 `outerhmm` full path에서 2 score-second 이상 relocation burst를 검출했다.
- jump 2초 뒤 위치에서 `accept`와 `hold` 손실을 비교했다.
- `accept`: relocation tracker가 2초 동안 계속 갱신한 최신 위치.
- `hold`: 사용자에게 보여줄 위치만 jump 직전 origin에 유지.
- 평가 손실: GT 위치와의 거리, 5초 cap.
- 8 development works 중 eligible event가 있는 7작품에 leave-one-work-out을 적용했다.

비교 모델:

1. train works에서 고른 단일 global action.
2. jump 순간 정보만 쓰는 immediate model.
3. 2초 뒤까지의 tracker 안정성·다른 tracker 동의·performance chroma의 origin/destination 지지를 쓰는 delayed model.

## 결과

- eligible events: 82.
- works: 7.
- 두 tracker 모두 포함.
- event oracle action: accept 74, hold 8.

| 모델 | work-equal regret | work-equal action accuracy |
|---|---:|---:|
| train-global | 0.248375 | 0.913605 |
| immediate | 0.248375 | 0.913605 |
| delayed 2s | 0.116434 | 0.953968 |

개선:

- `regret(immediate)-regret(delayed)`: +0.131941.
- work bootstrap 95%: `[0.012980, 0.264167]`.
- global 대비 개선도 같은 값과 CI.
- 작품별: 엄격 개선 3, 동률 4, 악화 0.

immediate model이 global과 완전히 같았다는 것은 jump 크기·방향과 jump 순간 다른 tracker 위치만으로는 hold할 사건을 가려내지 못했다는 뜻이다. 실제 이득은 jump 후 2초의 동적 정보에서 나왔다.

## 작품별 구조

regret가 0보다 컸던 세 작품에서는 모두 delayed validator가 개선했다.

- Beethoven Sonata 28: 0.5782 → 0.1608.
- Beethoven Sonata 3: 0.9849 → 0.5695.
- Chopin Sonata 2: 0.1756 → 0.0848.

나머지 네 작품에서는 global/immediate regret이 이미 0이었고 delayed도 0을 유지했다.

이 때문에 평균 개선과 bootstrap CI는 통과했지만 “최소 5작품에서 엄격 개선” gate는 구조적으로 통과하지 못했다. 이 gate가 지나치게 보수적이었다는 평가는 가능하지만, 결과를 본 뒤 규칙을 바꿔 이번 실험을 성공으로 재분류하지 않는다.

## gate 판정

- event/works/두 tracker support: 통과.
- immediate 대비 평균·bootstrap 개선: 통과.
- global 대비 평균·bootstrap 개선: 통과.
- 60% works 엄격 개선: 실패(3/7, 요구 5/7).
- 전체: 실패, `lockbox_authorized=false`.

## 독립 감사

`scripts/audit_online_relocation_validation.py`가 모델을 독립 재적합했다.

- 82 events, 7 works 재현.
- 각 fold regret 최대 잔차 0.
- bootstrap CI 동일.
- 개선 3, 동률 4, 악화 0 재현.
- hash drift 0.
- lockbox output absent.
- 관련 단위 테스트 6개 통과.

## 무엇을 결론낼 수 있나

현재 가장 실용적인 제한 결론은 다음이다.

> 큰 relocation은 즉시 맞다/틀리다를 판단하기 어렵지만, 2초의 후속 tracker·관측 정보를 이용하면 잘못된 accept 결정의 손실을 줄일 가능성이 있다.

이는 모호성 기반 제어보다 강한 개발 신호다. 하지만 8곡 중 eligible 7곡, hold oracle 8건뿐이고 동일 ASAP pipeline이다. 아직 일반적인 시스템 권고나 논문 핵심 확증으로 쓰면 안 된다.

## 다음 단계

현재 lockbox gate를 사후 수정하지 않는다. 다음 선택지는 두 가지다.

1. 새로운 development 작품을 추가해 “기존 regret=0인 작품에서는 비열화, regret>0인 작품에서는 개선”이라는 non-inferiority/conditional-benefit gate를 결과 전에 새로 고정한다.
2. 또는 현재 모델·feature·2초 horizon을 그대로 고정하고, lockbox 사용 전에 별도의 소규모 validation cohort를 확보한다.

같은 7작품에서 delay, feature, ridge, jump threshold를 조정하면 안 된다.

## 산출물

- `scripts/online_relocation_validation.py`
- `scripts/audit_online_relocation_validation.py`
- `results/online_relocation_validation/result.json`
- `results/online_relocation_validation/events.json`
- `results/online_relocation_validation/independent_audit.json`
