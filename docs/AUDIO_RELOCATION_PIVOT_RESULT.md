# Audio relocation rescue와 선행연구 감사 최종 결과

작성일: 2026-09-24  
최종 판정: **audio rescue FAIL / delayed-decision 원리의 최초성 없음**

## 가장 중요한 결론

두 가지 가능성을 분리해 검사했다.

1. frozen MIDI ridge만 audio에 전이되지 않은 것인가?
2. 아니면 `>=2 score-second relocation 뒤 후보를 검증`하는 문제 설정 자체가 현재 audio tracker에 맞지 않는가?

미사용 Vienna 32연주에서 parameter-free audio mean-evidence 규칙까지 prospective하게 검증한 결과, 두 번째 문제가 더 컸다. 사건은 여전히 한 tracker와 두 작품에만 희소했고, 단순 audio evidence는 baseline보다 regret을 악화시켰다.

## 선행연구 중복 판정

현재 아이디어의 중심인 “여러 위치 후보를 추가 audio 문맥으로 검증하고 accept/reject/postpone한다”는 이미 존재한다.

- Arzt & Widmer (2010)는 모든 악보 위치에서 후보를 만들고 여러 ODTW matcher로 병렬 검증하며 `accept/reject/postpone`했다. 최소 5초의 grounding과 self-similarity structure model도 사용했다.
- Yamamoto et al. (2013)은 미래 관측으로 과거 위치를 정하는 delayed-decision Viterbi와 현재 위치 anticipation을 제안했다.
- Fox et al. (2007)은 bottom-up 관측으로 새 위치 가설을 주입해 lost tracker를 회복했다.
- Nakamura et al. (2015/2016)은 arbitrary repeat/skip과 오류를 포함하는 실시간 HMM을 실제 연습 연주에서 평가했다.

따라서 본 작업을 `최초의 delayed validation`, `최초의 multiple-hypothesis recovery`, `최초의 accept/hold`로 주장하면 안 된다. 상세 문헌과 링크는 `AUDIO_RELOCATION_NOVELTY_AND_PIVOT_20260924.md`에 기록했다.

남을 수 있는 차별점은 좁다.

- 기존 tracker를 수정하지 않는 post-hoc wrapper
- alignment accuracy 대신 행동 regret 평가
- frozen MIDI→audio transfer와 독립 lockbox 감사
- static score ambiguity와 observation-dependent candidate evidence의 분리

이는 평가 설계의 차별점이지 delayed-decision 알고리즘 원리의 신규성이 아니다.

## 열린 5건에서의 방법 변경 진단

Vienna 최초 8연주의 열린 5 event에서 세 규칙을 비교했다.

| 규칙 | 정확 | hold | oracle hold TP | 평균 regret |
|---|---:|---:|---:|---:|
| 시간평균 chroma 후보 비교 | 4/5 | 1 | 1 | 0.3787 |
| frame-order diagonal | 3/5 | 0 | 0 | 1.0414 |
| local DTW | 3/5 | 0 | 0 | 1.0414 |

시간 순서를 보존하거나 DTW를 쓰면 더 좋아질 것이라는 가설은 즉시 지지되지 않았다. 평균-chroma 규칙만 탐색적으로 좋아 보여 이를 새 표본에 동결했다. 이 5건은 이미 outcome을 본 뒤의 탐색이므로 성공 증거로 세지 않았다.

## prospective 32연주 결과

결과를 보지 않고 고정한 performer `p02,p05,p08,p11,p14,p17,p20,p22` × Vienna 4작품을 실행했다.

### 구현

- 32연주 × 2 tracker path validity 통과
- GT anchor 112–201개, 엄격 단조
- freeze/development hash drift 0

### support

| 항목 | 결과 | 사전 기준 |
|---|---:|---:|
| eligible event | 9 | ≥20 |
| event 연주 | 7 | ≥12 |
| event 작품 | 2 | ≥3 |
| event tracker | `outerhmm`만 | 둘 다 |

작품별로 Chopin Op.38 1건, Schubert D.783 No.15 8건이었다. Chopin Op.10 No.3와 Mozart K.331에는 0건이었다. `arzt`는 32연주 전체에서 0건이었다.

최초 8연주까지 합치면 Vienna 40연주에서 eligible 14건이며, 모두 `outerhmm`, 같은 두 작품에만 있었다. 이 사후 합산은 효과 검정이 아니라 trigger coverage 진단이다.

### 행동효용

- oracle: hold 4건, accept 5건
- mean-evidence: hold 5건
- hold precision: `3/5 = 0.60` (기준 ≥0.80)
- frozen delayed regret는 global accept와 동일
- work-equal global−mean improvement: `−0.23492`
- bootstrap 95%: `[−0.46984, 0.0]`
- Schubert regret: global `0.15325` → mean-evidence `0.62309`
- Chopin Op.38: 둘 다 `2.5`

열린 5건에서 보였던 mean-evidence의 4/5 결과는 재현되지 않았다. parameter-free audio evidence는 false hold를 만들어 오히려 손실을 키웠다.

## 무엇이 실패했나

단순히 학습된 ridge만 실패한 것이 아니다.

1. **trigger coverage 실패**: 큰 relocation이 `outerhmm` 한 방법과 일부 작품에만 생긴다.
2. **candidate evidence 실패**: 평균 chroma, ordered diagonal, local DTW 모두 안정적인 accept/hold 규칙이 되지 못했다.
3. **새로움 제한**: 후보를 기다렸다가 검증하는 구조 자체도 기존 연구와 겹친다.

따라서 같은 Vienna에서 jump threshold, delay, similarity margin, DTW 제약을 바꾸는 rescue는 중단한다. 이는 실패를 만회하는 검증이 아니라 outcome에 맞춘 방법 탐색이 된다.

## 논문에 미치는 영향

MIDI lockbox 결과는 계산적으로 유지된다. 그러나 그 결과를 새로운 delayed-decision score follower라고 포장할 수는 없다. 가능한 논문 방향은 알고리즘 최초성보다 다음의 empirical finding이다.

> Static score ambiguity did not provide an operational trigger; event-conditioned delayed validation reduced decision regret on unseen symbolic performances, but both the event support and the decision boundary failed to transfer to real audio trackers. Existing delayed-decision and multi-hypothesis methods therefore require explicit domain- and tracker-specific coverage evaluation before being treated as safety mechanisms.

LBD에서는 `decision-utility evaluation + cross-domain failure analysis`로는 가능성이 있다. 새 방법 full paper의 중심 결과로는 부족하다.

## 산출물

- 사전계약: `AUDIO_MEAN_EVIDENCE_VALIDATION_PROTOCOL.md`
- 실행: `scripts/online_relocation_audio_mean_evidence.py`
- 독립감사: `scripts/audit_online_relocation_audio_mean_evidence.py`
- 결과: `results/online_relocation_audio_mean_evidence/result.json`
- 감사: `results/online_relocation_audio_mean_evidence/independent_audit.json`
- 열린 진단: `results/audio_relocation_continuation_diagnostic/result.json`
- 감사 최대수치잔차 0, 관련 6 tests PASS, 4 scripts compile PASS

