# 2초 relocation validator lockbox 확증 — 동결 계약

작성일: 2026-09-24. lockbox MIDI/annotation outcome을 실행·열람하기 전에 고정한다.

## 목적과 투명한 변경

development에서 delayed validator는 regret `.2484 -> .1164`, 개선 bootstrap95 `[.0130,.2642]`, 개선3/동률4/악화0을 보였지만 “5/7 strict improvement” gate는 실패했다. 네 작품의 baseline regret이 이미 0이라 strict improvement가 불가능했다는 사실을 본 뒤 이 lockbox 규칙을 설계했다.

따라서 이것은 기존 실패 gate를 소급 통과시키는 것이 아니다. 동일 모델을 새 자료에서 검증하되, floor effect를 사전에 처리하는 **새 확증 계약**이다.

## 완전 동결 항목

- tracker pair: `arzt`, `outerhmm`.
- Matchmaker commit: `40fd6e7602bd9e596cea1beafdcee4853b9f2c98`.
- jump: 연속 full-path score 변화 절댓값 >=2초, performance time >=10초, 2초 이내 burst에서 최대 jump.
- validation delay: 2.0초.
- accept/hold 정의, 5초 capped loss, immediate/delayed feature 순서와 계산.
- development 82 events로 fit한 train-work-equal ridge alpha=1 모델.
- 예측 threshold 0. 모델 예측 target>=0이면 accept, 아니면 hold.
- development global action은 `accept`로 동결.
- 위 항목은 lockbox 결과 후 변경하지 않는다.

## lockbox

- `output/ambiguity_map_fresh_cohort/manifest.csv`에서 사전에 보존한 `fl000`–`fl007` 8작품/8연주.
- development와 work family·performer overlap 0인 기존 split을 그대로 사용한다.
- 동일 ASAP annotation pipeline이므로 외부 corpus 독립확증은 아니다.

## 비교와 지표

- `global`: 모든 event에서 frozen development action `accept`.
- `immediate`: frozen development immediate model.
- `delayed`: frozen development delayed-2s model.
- event regret은 선택 loss minus event oracle loss.
- 먼저 work 안에서 평균하고 8 works를 동일 비중으로 평균한다.
- work bootstrap 20,000회, seed 20260924.

## floor-aware 확증 gate

모두 만족해야 확증으로 판정한다.

1. eligible events >=40, works >=6, 두 tracker 모두 포함.
2. `regret(global)-regret(delayed)`와 `regret(immediate)-regret(delayed)`의 work 평균 및 bootstrap95 하한이 모두 >0.
3. global regret가 0보다 큰 actionable work가 최소3개이고, 그중 60% 이상에서 delayed regret가 global보다 엄격히 작다.
4. global regret가 0인 floor work 각각에서 delayed regret <=0.05 score-second.
5. 전체 work 중 delayed가 global보다 0.05초 넘게 악화된 work가 0개.

accuracy는 보조 지표이며 gate가 아니다. gate 일부만 통과한 결과를 성공으로 쓰지 않는다.

## 해석

- 통과: 동일 ASAP pipeline의 미사용 작품에서 2초 validator의 accept/hold 행동효용을 확증한다. 실제 audio/device 실험과 외부 corpus 재현은 여전히 필요하다.
- 실패: development 개선은 해당 표본에 한정된 것으로 보고 이 validator를 실용 결론으로 승격하지 않는다. lockbox에서 threshold·delay·feature를 조정하지 않는다.
