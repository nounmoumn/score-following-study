# 데이터 — 취득 방법과 재현 방법

## 1. ASAP (공개, 자동으로 받아짐)

```
git clone --depth 1 https://github.com/fosfrancesco/asap-dataset.git ismir2026_lbd/asap_dataset
```

라이선스·출처: 저장소 자체 참조 (fosfrancesco/asap-dataset). 버전은 clone 시점의 최신 커밋 —
정확한 재현이 필요하면 `git -C ismir2026_lbd/asap_dataset rev-parse HEAD`로 커밋 해시를 같이 기록할 것.

## 2. week4 (비공개 — 개인 녹음, 아직 공유 준비 안 됨)

`bansanka_previous_guitar_take_recovered.wav`, `dont_look_back_in_anger_last_guitar_take.wav`는
사용자 본인의 기타 연주 녹음이다(§WEEK4_AUDIO_INVENTORY.md 참조). 저장소에 안 들어있고
개인 컴퓨터의 `kaist_madcamp/week4/` 폴더에만 존재한다.

**로컬에서 실행하려면**:
1. `data/local_paths.example.yaml`을 `data/local_paths.yaml`로 복사(이미 이 컴퓨터엔 돼있음)
2. `week4_root`를 본인 week4 프로젝트 경로로 수정
3. `configs/paper/p1_week4_*.yaml`은 `${week4_root}/...` 형태로 경로를 참조하므로 자동으로 치환됨

**체크섬**: `data/manifests/week4.csv`에 사용된 정확한 파일의 SHA-256이 기록돼 있다. 파일이
바뀌면(덮어써지면) 해시가 달라지므로, "그때 그 파일이 맞는지"를 나중에도 검증할 수 있다.

**공개 가능성 (아직 미결정, CAMPAIGN_LOG.md 참조)**:
1. 밴드/본인 동의 받아 원본 공개
2. 요청 시 제공 가능한 제한 데이터로 명시
3. ⭐ **당장 가능한 대안**: 원본 오디오 대신, `results/`에 남는 파생 산출물(크로마 특징,
   GT 매핑, 프레임별 오차·모호성 CSV)만 공개 — 원본 음원 저작권/사생활 문제 없이
   핵심 수치는 재현 가능해짐. 논문 제출 전 결정 필요.
4. 새로 수집하는 공개 가능 녹음(CAMPAIGN_LOG.md)으로 대체/보강

## 3. 데이터 제외 기록

현재 제외된 표본은 없음(진행 중인 실험이 얼마 안 됨). ASAP 파일럿에서 "건강도 필터"로
제외된 곡은 코드가 아니라 **실행 결과**이므로 `results/<experiment>/metrics.json`의
`unhealthy_pieces` 필드에 실행마다 자동 기록된다(고정 목록이 아님 — config가 바뀌면 달라질 수 있음).
