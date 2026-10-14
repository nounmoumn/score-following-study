import numpy as np

from score_following.online_dtw import OnlineDTW


def test_identical_sequences_give_diagonal_path():
    """같은 두 시퀀스를 넣으면 t==j인 대각선 경로가 나와야 한다."""
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(30, 12))
    seq = seq / np.linalg.norm(seq, axis=1, keepdims=True)

    dtw = OnlineDTW(seq, seq, band_width=5, max_run_count=3)
    path = dtw.run()

    est_j_of_t = {}
    for t, j in path:
        est_j_of_t[t] = j

    diffs = [abs(t - est_j_of_t[t]) for t in est_j_of_t]
    assert max(diffs) <= 1, f"대각선에서 너무 벗어남: max diff={max(diffs)}"


def test_real_path_drops_forced_tail():
    """연주가 스코어보다 짧은 접두어일 때, real_path()는 연주 종료 시점에서 끊겨야 한다
    (ENGINEERING_LOG.md의 '부수 발견' — 인위적 꼬리 제거)."""
    rng = np.random.default_rng(1)
    perf = rng.normal(size=(10, 12))
    perf = perf / np.linalg.norm(perf, axis=1, keepdims=True)
    score = rng.normal(size=(40, 12))
    score = score / np.linalg.norm(score, axis=1, keepdims=True)
    score[:10] = perf  # 연주와 스코어 앞부분을 일치시켜 최소한의 정합을 보장

    dtw = OnlineDTW(perf, score, band_width=5, max_run_count=3)
    dtw.run()
    real = dtw.real_path()

    last_t = real[-1][0]
    assert last_t == len(perf) - 1
    # 전체 경로(꼬리 포함)가 real_path보다 길거나 같아야 한다
    assert len(dtw.path) >= len(real)
