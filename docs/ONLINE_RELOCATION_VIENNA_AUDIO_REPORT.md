# 동결 relocation validator의 실제 오디오 전이 결과

작성일: 2026-09-24  
판정: **FAIL — 구현은 정상이나, 지원량과 행동효용 모두 확증하지 못함**

## 한 문장 결론

ASAP MIDI에서 확증된 “큰 재위치 뒤 2초를 기다려 accept/hold를 다시 판단한다”는 정책은 그대로 보존되지만, 현재 Matchmaker 기본 오디오 추적기와 Vienna 실제 피아노 녹음에 옮기면 **사건이 한 추적기에만 드물게 생겼고, 동결 모델은 정답이 hold인 2건까지 전부 accept하여 baseline을 개선하지 못했다.** 따라서 지금 상태로 실제 오디오 기능에 쓰면 안 된다.

## 무엇을 동결하고 검사했나

- 모델 SHA256: `233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff`
- 추적기: Matchmaker commit `40fd6e7602bd9e596cea1beafdcee4853b9f2c98`, `arzt`와 `outerhmm` 기본 설정
- 발동 조건: 연주 시작 10초 뒤, 한 번에 2 score-second 이상 이동한 burst
- 행동: 즉시 점프 수용(`accept`) 대 점프 전 위치 유지(`hold`)
- 판단 시점: 점프 2초 뒤
- 입력 특징, ridge 계수, threshold 0, 손실 cap 5초를 변경하지 않음
- Vienna 4작품 전부 × 결과를 보지 않고 고른 `p01`, `p12` = 8연주
- 결과 전에 [ONLINE_RELOCATION_VIENNA_AUDIO_PROTOCOL.md](ONLINE_RELOCATION_VIENNA_AUDIO_PROTOCOL.md)를 고정함

## 결과

### 1. 구현은 통과했다

- 8연주 × 2 추적기의 경로는 유한하고 시간 단조이며 악보 범위 안이었다.
- 작품별 `p01`의 20초 prefix 재실행은 full run의 안전구간과 일치했다.
- GT anchor는 연주당 112–201개였고 모두 엄격 단조였다.
- 따라서 결과 부족을 경로 파싱, 미래정보 누출, GT 붕괴로 설명할 수 없다.

### 2. 발동 사례가 사전 기준보다 훨씬 적었다

| 항목 | 관측 | 사전 기준 |
|---|---:|---:|
| eligible event | 5 | ≥20 |
| event가 있는 연주 | 3 | ≥6 |
| event가 있는 작품 | 2 | ≥3 |
| event가 있는 추적기 | `outerhmm`만 | 두 추적기 모두 |

원시 큰 점프는 6건이었다. 5건은 그대로 eligible이 되었고, Chopin Op.10 No.3의 1건만 곡 끝과 너무 가까워 2초 뒤 평가가 불가능했다. 즉, 후처리 필터가 많은 사건을 버린 것이 아니다. `arzt`에는 원시 큰 점프 자체가 0건이었다.

작품·연주별 event 수는 다음과 같다.

- Chopin Op.38: `p01=0`, `p12=1`
- Chopin Op.10 No.3: `p01=0`, `p12=0`
- Mozart K.331 1악장: `p01=0`, `p12=0`
- Schubert D.783 No.15: `p01=2`, `p12=2`

약 719.45초(0.200시간)의 오디오에서 5건이므로 단순 노출률은 시간당 25.0건이다. 그러나 한 추적기·두 작품에 몰렸기 때문에 이 값을 일반적인 발생률로 외삽하면 안 된다.

### 3. 적은 5건 안에서도 모델은 baseline을 개선하지 못했다

- 정답 행동: `accept` 3건, `hold` 2건
- 동결 모델 행동: `accept` 5건, `hold` 0건
- global baseline도 항상 `accept`이므로 두 정책은 완전히 같았다.
- work-equal regret: global `1.36086`, delayed `1.36086`
- improvement: `0.0`, bootstrap 95% 구간 `[0.0, 0.0]`
- event accuracy: `3/5 = 0.60`
- delayed 예측값 범위는 `+2.492`–`+3.710`으로 threshold 0에서 멀었다. 우연히 경계 바로 위여서 hold를 놓친 것이 아니다.
- 각 event의 delayed 특징은 development 평균에서 최대 `1.48` scale 이내였다. 단순한 극단값은 아니지만, 이것만으로 domain shift가 없다고 말할 수는 없다.

## 왜 FAIL인가

두 이유를 분리해야 한다.

1. **확증할 표본이 부족하다.** 사전 support gate를 통과하지 못해 평균효과나 hold precision을 신뢰성 있게 추정할 수 없다.
2. **관측된 사건에서는 유용한 행동 변화가 없었다.** 5건 모두 accept하여 baseline과 동일했고, 그중 hold가 더 나은 2건을 모두 놓쳤다.

따라서 “사건만 더 모으면 이미 성공한 모델”이라고 해석할 수 없다. 반대로 5건만으로 모델의 모든 실제 오디오 가능성을 반증했다고 말해도 안 된다. 정확한 판정은 **현재 frozen MIDI 정책의 오디오 전이 미확증이며, 이 작은 표본에서는 행동효용 신호가 0이었다**이다.

## 모호성 연구에 남는 결론

이번 정책이 사용하는 모호성은 정적 악보 유사도가 아니라, 재위치 뒤 두 후보가 실제 관측 오디오와 두 추적기 상태에 의해 어느 쪽으로 해소되는지를 나타내는 observation-dependent ambiguity다. 이 연결은 ASAP MIDI lockbox에서 regret을 `0.7003 → 0.3344`로 낮추며 확증되었다.

하지만 실제 오디오에서는 이 특징들이 단 한 번도 hold 행동을 발동하지 않았다. 그러므로 현재 주장 범위는 다음처럼 제한해야 한다.

> 큰 재위치 뒤 관측 증거로 후보를 재검증하는 방식은 동일 ASAP MIDI pipeline에서는 유용했다. 그러나 실제 오디오에서는 재위치의 발생 방식이 추적기별로 비대칭이고 frozen 결정경계가 필요한 hold를 발동하지 않아, 실사용 전이는 아직 성립하지 않았다.

정적 모호성 지도를 자동 제어 신호로 되살렸다는 결론은 아니다. 현재 지도는 실패 사건을 “두 후보 중 무엇이 관측과 맞았는가”로 감사하는 분석 축으로만 남는다.

## 실용적 다음 단계

같은 8개에서 threshold, delay, jump 크기나 특징을 사후 조정하지 않는다. 현 상태에서 더 많은 오디오만 무작정 반복하는 것도 우선순위가 아니다. 다음 실험은 별도의 audio-development 자료에서 아래 두 문제를 먼저 해결해야 한다.

1. tracker 고유의 큰 점프에만 의존하지 않는 audio-native 후보 충돌/불안정 trigger를 정의한다.
2. 그 trigger에서 실제 hold 사건을 충분히 확보해 decision boundary를 학습한 뒤, 작품 단위로 분리된 미사용 audio에서 검증한다.

최소 조건은 두 추적기 모두의 support, 최소 20 hold/accept 평가 사건, 미사용 작품, 사전 고정 threshold다. 이를 만족할 새 corpus나 분할이 없으면 실제 UI 자동제어로 진행하지 않는다. 이번 Vienna 8연주는 이미 결과가 열렸으므로 새 확증 lockbox로 재사용하지 않는다.

## 감사 및 재현

- 실행: `scripts/online_relocation_vienna_audio.py`
- 독립 감사: `scripts/audit_online_relocation_vienna_audio.py`
- 결과: `results/online_relocation_vienna_audio/result.json`
- 감사 결과: `results/online_relocation_vienna_audio/independent_audit.json`
- 독립 재계산 최대 수치 잔차 `0`, freeze/development hash drift `0`
- 관련 단위시험 `6/6 PASS`, 세 스크립트 compile PASS

앞선 공식 Matchmaker 2작품 pilot도 구현 gate는 통과했지만 eligible event가 0이었다. Vienna 확장은 그 “평가 불가능” 상태에서 5건을 얻었으나, support와 행동효용 모두를 통과시키지는 못했다.
