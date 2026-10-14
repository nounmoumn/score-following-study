# 동결 relocation validator — Vienna 4x22 audio shadow 계약

작성일: 2026-09-24. Vienna audio tracker outcome 계산 전에 고정한다.

## 목적

공식 Matchmaker 2작품 pilot은 구현·인과성은 통과했지만 relocation 0건으로 효용을 평가하지 못했다. 모델이나 jump threshold를 바꾸지 않고, synchronized audio/MIDI/score match가 있는 Vienna 4x22에서 자연 audio relocation support와 frozen validator 전이를 검사한다.

## outcome-blind 표본

- 4작품 전부: Chopin Op.38, Chopin Op.10 No.3, Mozart K.331 1악장, Schubert D.783 No.15.
- 각 작품에서 고정 performer `p01`, `p12`: 총 8연주.
- 선택은 작품 전체 포함과 performer index의 간격 표본이며 tracker 결과를 사용하지 않는다.
- 공식 audio ZIP, local pinned match/MIDI/score 파일을 사용한다.

이 corpus는 이전 P3 연구에서 다른 endpoint가 열린 자료이므로 pristine 독립자료는 아니다. 그러나 현재 validator와 audio tracker outcome은 열기 전에 고정하며 ASAP 개발/lockbox와 corpus가 다르다.

## 동결 요소

- frozen model SHA256 `233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff`.
- Matchmaker commit `40fd6e7602bd9e596cea1beafdcee4853b9f2c98`, audio `arzt`와 `outerhmm` 기본 설정.
- jump >=2 score-second, onset >=10초, 2초 burst 최대 jump, delay 2초.
- accept/hold, cap 5초, feature 순서, threshold 0 변경 없음.
- score GT는 official `.match`의 matched note를 score onset별로 묶은 performance-time median과 score MIDI second를 사용한다. 미래 tracker output은 feature에 사용하지 않는다.

## 구현 gate

- 8연주×2 tracker path 유한·performance-time 단조·score-domain 내부.
- 각 작품 `p01`의 20초 prefix가 full path의 19.75초 이전과 최대잔차 `1e-9` 이하.
- GT time/score position 유한·엄격 단조, 각 연주 최소20 anchor.

## 확증 전이 gate

모두 만족해야 `PASS`다.

1. 구현 gate 전체 통과.
2. eligible events >=20, performances >=6, works >=3, 두 tracker 모두 포함.
3. work-equal `global regret - delayed regret >0`.
4. 4-work bootstrap 20,000회(seed 20260924)의 improvement 하한 >0. 표본이 작으므로 보조 gate이며 결과에 정확 randomization도 함께 보고한다.
5. 어느 작품도 delayed regret가 global보다 0.05초 넘게 악화되지 않는다.
6. delayed hold 최소5건, hold precision >=0.8.

실패하면 delay/threshold/jump/feature를 재조정하지 않는다. support가 부족하면 자연 audio에서 큰 relocation이 드물다는 applicability 한계로 판정한다. 통과해도 실제 microphone/device latency 검증은 별도다.

