# 2초 온라인 relocation validator lockbox 확증 보고서

작성일: 2026-09-24  
사전 계약: `ONLINE_RELOCATION_LOCKBOX_PROTOCOL.md`

## 결론

동결된 2초 validator는 보존해 둔 lockbox에서 **사전 floor-aware gate를 모두 통과했다.** 같은 ASAP MIDI/annotation pipeline의 미사용 작품에 대해서는, 큰 위치 jump를 무조건 즉시 받아들이는 것보다 2초간 후속 증거를 확인한 뒤 accept/hold를 결정하는 편이 손실을 줄인다는 결론이 재현됐다.

핵심은 static score ambiguity map이 아니다. 유효했던 정보는 jump 뒤 두 tracker가 어디로 안정되는지, 서로 동의하는지, 실제 연주 chroma가 원래 위치와 새 위치 중 어디를 지지하는지였다. 즉 모호성은 “악보에 비슷한 곳이 존재한다”는 정적 점수보다 **현재 관측으로 두 위치 후보가 아직 구별되지 않았는가, 2초 뒤 어느 쪽으로 해소됐는가**라는 동적 검증 문제로 살릴 수 있다.

## 동결 lockbox 결과

- 입력: `fl000`–`fl007`, 8연주/8작품. 그중 7작품에서 eligible relocation 61건; `fl001`은 0건.
- tracker: `arzt`, `outerhmm`.
- 개발/lockbox 작품 및 연주자 중복: 0.
- work-equal regret: global/immediate `0.700297` → delayed `0.334358`.
- work-equal improvement: `+0.365939` score-second.
- work bootstrap 95%: `[0.033115, 0.823535]`.
- work-equal action accuracy: `71.01%` → `83.40%`.
- actionable works: 5; 개선 4, 동률 1. 요구치는 3개 개선.
- floor works: 2; 둘 다 delayed regret 0.
- 0.05초보다 크게 악화된 작품: 0.
- 사전 6개 gate: 모두 통과, 최종 `PASS`.

immediate model은 61건을 전부 accept해 global baseline과 같았다. delayed model은 5건만 hold했고, 이 5건은 모두 hold가 실제로 더 나은 사건이었다. oracle hold 9건 중 5건을 잡았으므로 이 표본에서 hold precision은 100%, recall은 55.6%였다. 표본이 작으므로 이 두 비율 자체를 일반 성능치로 간주하지 않는다.

## 추가 압박 검증

사전 gate를 바꾸지 않은 채 결과의 취약성을 사후 진단했다.

- event-weighted regret: `0.385671 → 0.173563`, 개선 `+0.212108`.
- 일곱 작품을 하나씩 제외한 평균 개선 범위: `+0.158265` 이상. 한 작품에만 의존하지 않았다.
- 작품별 개선값은 `[0, 0.2318, 0, 1.6120, 0.6970, 0.0208, 0]`; 악화는 없었다.
- 다만 0이 아닌 네 작품만 놓은 exact one-sided sign-flip 검정은 `p=0.0625`였다. bootstrap CI는 양수지만, 작품 수가 작아 관습적 0.05 기준의 강한 비모수 유의성까지 확보한 것은 아니다.

## 독립 감사

`scripts/audit_online_relocation_lockbox.py`가 실행 코드를 import하지 않고 frozen affine predictor를 다시 구현해 다음을 재계산했다.

- 8개 case event 파일과 aggregate 61 events의 일치.
- frozen model 예측, accept/hold 행동, regret, accuracy, 작품 평균, bootstrap 및 전체 gate.
- 최대 수치 잔차 0.
- lockbox 입력 hash drift 0, 개발 model source hash drift 0.
- 개발/lockbox work·performer overlap 0.
- 관련 인과성 단위 테스트 6개와 Python compile 통과.

감사 결과는 `results/online_relocation_validation_lockbox/independent_audit.json`에 저장했다.

## 실용적 결론

현재 가장 방어 가능한 행동 규칙은 다음이다.

> 온라인 추적기가 2 score-second 이상 크게 재위치하면 곧바로 사용자 표시를 확정하지 말고, 2초간 추적기 안정성·추적기 간 동의·연주 관측의 후보 지지를 모은다. 동결 validator가 reject할 때만 기존 위치를 유지하고, 나머지는 새 위치를 받아들인다.

이는 “모호성이 높으면 항상 기다린다”는 규칙이 아니다. relocation이 만든 두 경쟁 후보 사이의 **관측 기반 해소 여부**를 행동에 연결한다. 따라서 기존 static ambiguity map은 자동 제어 입력이 아니라 후보 문맥 설명·사후 감사 자료로 남기고, 실시간 행동은 동적 validator가 담당하는 구성이 맞다.

## 제한과 다음 확증

- lockbox는 결과 미사용 작품/연주자였지만 같은 ASAP corpus, 같은 MIDI annotation pipeline이다. 외부 corpus 독립확증이 아니다.
- 실제 microphone audio, 장치 latency, 누락·오음, 실제 사용자 화면에서의 2초 지연은 검증하지 않았다.
- eligible 작품은 7개이고 한 작품은 사건 0건, 두 작품은 사건 1건뿐이다.
- 이 실행으로 `fl000`–`fl007` lockbox는 처음 개봉되었으며 더는 pristine lockbox가 아니다. 이 자료에서 delay, threshold, feature를 재튜닝하면 안 된다.
- 다음 단계는 모델을 그대로 고정한 외부/마이크 오디오 shadow test다. 우선 사용자 표시를 실제 변경하지 않고, relocation 발생 시 validator 판단과 이후 정답만 기록해 false hold와 놓친 hold를 측정한다.

## 산출물

- `ONLINE_RELOCATION_LOCKBOX_PROTOCOL.md`
- `output/online_relocation_validator_frozen/model.json`
- `scripts/online_relocation_lockbox.py`
- `scripts/audit_online_relocation_lockbox.py`
- `results/online_relocation_validation_lockbox/result.json`
- `results/online_relocation_validation_lockbox/events.json`
- `results/online_relocation_validation_lockbox/independent_audit.json`

