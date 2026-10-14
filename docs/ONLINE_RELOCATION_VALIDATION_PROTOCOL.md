# 온라인 relocation 검증 — 사전 고정 계약

작성일: 2026-09-24. validation 결과 계산 전에 고정한다.

## 실용 질문

`arzt` 또는 `outerhmm`이 2 score-second 이상 jump했을 때 즉시 새 위치를 신뢰하지 않고 2초 동안 후속 관측을 본 뒤, 새 위치를 수용(`accept`)할지 jump 전 위치를 출력으로 유지(`hold`)할지 더 잘 결정할 수 있는가?

목표는 jump 원인 설명이 아니라 위치 피드백 손실 감소다.

## 자료와 event

- 열린 development 8작품의 두 tracker full path와 ASAP alignment만 사용한다.
- 각 tracker에서 연속 출력 score 변화 절댓값 >=2초, performance time >=10초.
- 2초 이내 jump burst는 가장 큰 jump 하나로 합친다.
- event+2초가 tracker path와 GT annotation 범위 안에 있어야 한다.
- 같은 event 정의를 결과 후 변경하지 않는다.

## 행동과 손실

event time을 `t0`, 검증 시점을 `t1=t0+2초`로 한다.

- `accept`: relocation tracker가 `t1`까지 실제로 갱신한 최신 위치를 출력한다.
- `hold`: 내부 tracker는 계속 돌리되 사용자 출력만 jump 직전 origin에 고정한다.
- loss: `min(abs(output score-second - GT(t1)), 5초)`.
- oracle은 event마다 accept/hold 중 작은 loss다.
- 모델 target은 `loss(hold)-loss(accept)`. 양수면 accept가 낫다.

## 비교기

모두 leave-one-work-out이며 held-out GT를 fit이나 행동 선택에 쓰지 않는다.

1. `train_global`: train works의 평균 target 부호로 모든 held-out event에서 같은 행동.
2. `immediate`: t0까지 이용 가능한 jump 크기·방향·score progress와 다른 tracker의 origin/destination 근접도.
3. `delayed_2s`: immediate에 t1까지의 relocation 안정성, 두 tracker 합의 변화, performance 2초 chroma가 origin/destination score 2초 문맥을 지지하는 정도를 추가.

두 회귀 모두 train-work equal weight, train-only 표준화, ridge alpha=1. threshold는 예측 target 0으로 고정한다.

## causal feature

- score는 미리 알려져 있으므로 origin/destination score chroma를 읽을 수 있다.
- performance observation은 `[t0,t1]`만 사용한다.
- tracker position은 해당 시점 이하의 마지막 출력만 사용한다.
- 미래 GT는 평가에만 사용한다.

## development gate

다음을 모두 만족해야 frozen delayed validator를 lockbox 확인 후보로 올린다.

1. eligible event >=40, works >=6, 두 tracker 모두 포함.
2. work-equal `regret(immediate)-regret(delayed)` 평균 >0이고 work bootstrap95 하한 >0.
3. work-equal `regret(train_global)-regret(delayed)` 평균 >0이고 bootstrap95 하한 >0.
4. delayed regret가 immediate보다 작은 work와 train_global보다 작은 work가 각각 60% 이상.

항목 하나라도 실패하면 2초 validation의 추가 행동효용은 미확인이다. accuracy만 따로 성공으로 쓰지 않는다.

## 해석 제한

- `hold`는 2초간 사용자 출력 고정의 counterfactual이며 tracker 내부 상태 rewind가 아니다.
- ASAP MIDI/annotation 기반이고 실제 device/audio latency 실험이 아니다.
- 통과하더라도 development 결과이며 별도 lockbox 확인 전 일반화하지 않는다.
