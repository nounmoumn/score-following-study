import numpy as np

from score_following.offline_dtw import OfflineDTW


def _normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_identical_sequences_give_diagonal_path():
    """같은 두 시퀀스를 넣으면(같은 길이) t==j인 대각선 경로가 나와야 한다.
    OnlineDTW의 동명 테스트와 짝 — 온라인/오프라인이 쉬운 케이스에서 같은 답을
    내는지 확인하는 최소 기준선."""
    rng = np.random.default_rng(0)
    seq = _normalize(rng.normal(size=(30, 12)))

    dtw = OfflineDTW(seq, seq, open_end=False)
    path = dtw.path()

    j_of_t = dict(path)
    diffs = [abs(t - j_of_t[t]) for t in j_of_t]
    assert max(diffs) == 0, f"동일 시퀀스인데 대각선을 벗어남: max diff={max(diffs)}"
    assert path[0] == (0, 0)
    assert path[-1] == (29, 29)


def test_open_end_finds_correct_interior_endpoint():
    """week4류 상황(연주가 스코어의 접두어일 뿐, 시작점은 이미 알고 있음 —
    onset-detection으로 검증됨): 시작은 (0,0)로 고정하되, 스코어가 연주보다 훨씬
    길고 연주는 스코어 앞부분과만 일치한다. open_end=True면 억지로 끝(n-1)까지
    끌려가지 않고 실제로 일치가 끝나는 지점 근처에서 멈춰야 한다."""
    rng = np.random.default_rng(1)
    perf = _normalize(rng.normal(size=(20, 12)))
    score = _normalize(rng.normal(size=(100, 12)))
    score[0:20] = perf  # 스코어의 앞부분이 연주 전체와 일치(연주=스코어의 접두어)

    dtw = OfflineDTW(perf, score, open_end=True)
    dtw.run()

    # 연주 마지막 프레임(t=19)은 스코어의 19번 프레임과 맞아떨어져야 한다
    assert dtw.end_j_ == 19, f"열린 끝점이 엉뚱한 곳에서 끝남: end_j={dtw.end_j_}"
    j_of_t = dtw.j_of_t()
    assert j_of_t[0] == 0
    assert j_of_t[19] == 19


def test_open_end_false_forces_full_score_coverage():
    """open_end=False(기본, 고전적 전역 DTW)면 짧은 연주라도 끝점이 무조건 (m-1, n-1)."""
    rng = np.random.default_rng(2)
    perf = _normalize(rng.normal(size=(20, 12)))
    score = _normalize(rng.normal(size=(100, 12)))
    score[0:20] = perf

    dtw = OfflineDTW(perf, score, open_end=False)
    dtw.run()
    assert dtw.end_j_ == 99
    assert dtw.path()[-1] == (19, 99)


def test_no_degenerate_stuck_solution():
    """회귀 테스트 — ENGINEERING_LOG.md "DTW 퇴화 버그" 절 재현.
    스코어에 '연주 전체와 평균적으로 가장 가까운' 단일 화음 하나를 심어두면,
    가중치 없는 DTW(무가중치 수평/수직/대각)는 그 지점에 눌러앉는 게 더 싸져서
    기울기 0인 퇴화 해를 낸다. OfflineDTW는 대각=2 가중치를 쓰므로 이 함정에
    빠지지 않고 실제 대각 정렬을 찾아야 한다."""
    rng = np.random.default_rng(3)
    m = 40
    perf = _normalize(rng.normal(size=(m, 12)))
    # 스코어: 연주와 정확히 일치하는 대각 정렬 구간 + '평균적으로 그럴듯한' 미끼 화음
    score = _normalize(rng.normal(size=(m, 12)))
    score[:m] = perf  # 완벽한 대각 정렬이 존재하도록
    decoy = perf.mean(axis=0)
    decoy = decoy / np.linalg.norm(decoy)
    score[m // 2] = decoy  # 중간 지점에 "평균적으로 다 비슷한" 미끼를 심음

    dtw = OfflineDTW(perf, score, open_end=False)
    path = dtw.path()
    j_of_t = dict(path)

    # 퇴화라면 대부분의 t가 같은 j(미끼 위치)로 쏠려서 기울기가 0에 가까워진다.
    js = [j_of_t[t] for t in sorted(j_of_t)]
    n_unique = len(set(js))
    assert n_unique > m // 2, (
        f"퇴화 의심 — 서로 다른 j값이 {n_unique}개뿐 (m={m}). "
        f"대부분의 t가 한 지점에 눌러앉았을 가능성."
    )
    # 대각선에서 크게 벗어나지 않아야 한다 (완벽한 대각 정렬이 실제로 존재하므로)
    diffs = [abs(t - j_of_t[t]) for t in j_of_t]
    assert np.median(diffs) <= 2, f"대각선에서 중앙값 {np.median(diffs)}만큼 벗어남"


def test_band_width_prevents_distant_repeat_lockon():
    """회귀 테스트 — 2026-09-08 bansanka(만찬가) 실행 중 실제로 발생한 퇴화 재현.

    open_end=True + band 없음일 때: 국소비용이 전반적으로 높은(=진짜 대응조차 유사도가
    낮은, week4 실제 오디오처럼) 상황에서 스코어 앞부분에 '연주 전체와 어중간하게 계속
    비슷한' 짧은 반복 구간이 있으면, 대각선을 타고 끝까지 진행하는 것보다 그 반복
    구간에 눌러앉는 게 총비용이 더 싸진다. band_width를 주면 그 반복 구간이 나중
    프레임에서는 |i-j|가 너무 커져 도달 불가능해지므로 이 퇴화를 막아야 한다.

    ⚠️ 노이즈 수준·길이가 작으면 이 퇴화가 재현 안 된다(디코이가 안 유혹적임) —
    아래 파라미터(m=600, noise=1.2)는 실제 bansanka와 비슷한 자릿수(대각선 평균
    코사인유사도 ~0.2, 실제는 ~0.5였지만 방향은 같음)로 맞춰 실제로 band 없이
    돌리면 실패하는 것을 직접 확인하고 고른 값."""
    rng = np.random.default_rng(4)
    m = n = 600
    perf = _normalize(rng.normal(size=(m, 12)))
    score = _normalize(rng.normal(size=(n, 12)))
    noise = 1.2 * rng.normal(size=(m, 12))
    score[:m] = _normalize(perf + noise)  # 진짜 대응(대각선)이지만 유사도가 낮음(~0.2)

    # 스코어 앞부분(0~9번 프레임)에 '연주와 어중간하게 계속 비슷한' 반복 구간을 심음
    decoy_center = perf[:10].mean(axis=0)
    decoy_center = decoy_center / np.linalg.norm(decoy_center)
    for j in range(10):
        score[j] = decoy_center

    # band 없이 돌리면 실제로 decoy(끝점이 m-1 근처가 아님)에 눌러앉는지 먼저 확인
    # — 이 회귀 테스트가 뭔가 진짜를 막고 있는지 자체 검증(아니면 테스트가 공허해짐)
    no_band = OfflineDTW(perf, score, open_end=True, band_width=None)
    no_band.run()
    assert no_band.end_j_ < m - 100, (
        "이 테스트 파라미터로는 band 없이도 decoy에 안 걸림 — 회귀 재현 실패, "
        "노이즈 수준을 다시 조정할 것"
    )

    band = 30  # 대각선(|i-j|<=band) 근처만 허용 — decoy(j<10)는 i>40부터 도달 불가능
    with_band = OfflineDTW(perf, score, open_end=True, band_width=band)
    with_band.run()
    assert with_band.end_j_ >= m - 1 - band, (
        f"band를 줬는데도 먼 곳(끝점={with_band.end_j_})으로 끌려감 — band 제약이 안 먹힘"
    )
