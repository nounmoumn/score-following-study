"""Record the completed frozen P3 state experiment in project handoff files."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p3_state_conditioned_20260918_003032'
entry='''
## 2026-09-18 사전 상태·관측 구별력 분리 검증 완료 — 세 주검정 미달

실행 전 [P3_STATE_PROTOCOL.md](P3_STATE_PROTOCOL.md)와
`configs/p3_state_conditioned.yaml`을 고정한 뒤 기존40 score 학습/새40 score 검증.
각480시점, 후보와 가중치는 노출 전에 고정하고 이후2초 관측만 평가했다.
목표는 노출완료 오차에서 이후5초의 log1p오차 증가. P1/P2 재검정은 하지 않았다.

| 5초 비교 | 예측 MAE 상대개선 | score 평균개선 95% CI | Holm p |
|---|---:|---|---:|
| 관측약화 추가 M1 vs M0 | -1.68% | [-.011488, -.002928] | 1.0000 |
| 사전경쟁×관측약화 포함 M2 vs M0 | -5.67% | [-.040278, -.007240] | 1.0000 |
| 노출완료오차 통제 M4 vs M3 | -3.54% | [-.044983, +.018834] | 1.0000 |

음의 개선은 예측 악화. 세 비교 모두 양의 개선 기준 미달이며 원래 자연 P3는 여전히 미검증.
2/10초 보조 결과와 작곡가 단위 결과까지 [P3_STATE_REPORT.md](P3_STATE_REPORT.md)에 공개.
새자료480시점 중 사전 경쟁비율R≥1%는46개, 건강한 시점406개.
이 수치를 근거로 사후 위험집합을 좁히거나 성공 시계만 선택하지 않았다.

**확정 범위**: 고정한 Bayes·ridge·고정속도 관측척도에서 추가예측력이 확인되지 않았다.
이를 모든 가능한 관측척도/추적기에서 P3가 거짓이라는 결론으로 확장하지 않는다.
GT oracle·전체 MIDI 길이 비율·같은 작품 악장/버전 상관·작곡가5명 표본 한계가 남는다.
W는 실제 filter의 모든 템포경로를 주변화한 가능도가 아닌 고정속도 후보의 구별력이다.
수학적 항등식과 인과적/경험적 P3 증명은 구분을 유지한다.

**검증**: `python -m pytest tests -q` → 57 passed.
재생80개 모두 기준 Bayes 경로와 일치, 기존40개 저장경로도 일치.
960행/1440예측·score 비중복·시간범위·유한값 확인.
원본: `results/p3_state_conditioned_20260918_003032` (manifest, provenance,
source_snapshot, cases, landmarks, predictions, score_losses, metrics).
실행: `python scripts/p3_state_conditioned.py` 및
`python scripts/report_p3_state.py --result results/p3_state_conditioned_20260918_003032`.

**다음 우선순위**: 새 자연자료를 계속 소모하기 전에, 이미 사용한 자료와 합성 양성대조에서
실제 Bayes transition 전체를 적분한 관측가능도 측정의 타당성을 확인한다.
별도 2×2 교란(위치 상태 교란 유/무 × 증거 소거 유/무)은 조건부 인과적 증폭을
검증하는 설계로 구분한다. 어떤 성공도 자연 발생 선후관계의 증명으로 대체하지 않는다.

'''
for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
    path=ROOT/name;old=path.read_text(encoding='utf-8')
    if '## 2026-09-18 사전 상태·관측 구별력 분리 검증 완료' not in old:
        first,sep,rest=old.partition('\n');path.write_text(first+'\n'+entry+rest,encoding='utf-8')
files=[ROOT/'scripts/report_p3_state.py',Path(__file__),ROOT/'P3_STATE_REPORT.md',ROOT/'tests/test_frozen_evidence.py']
for f in files:shutil.copy2(f,OUT/'source_snapshot'/f.name)
(OUT/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
print('Recorded final state experiment and report provenance.')
