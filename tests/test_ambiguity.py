import numpy as np

from score_following.ambiguity import (
    build_similarity_matrix,
    max_sim_ambiguity,
    cluster_entropy_ambiguity,
)


def _random_chroma(n, seed):
    rng = np.random.default_rng(seed)
    c = np.abs(rng.normal(size=(n, 12)))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def test_maxsim_higher_for_repeated_sequence_than_unique():
    window, exclude = 3, 4

    unique_chroma = _random_chroma(30, seed=0)
    sim_u, nw_u = build_similarity_matrix(unique_chroma, window)
    amb_u = max_sim_ambiguity(sim_u, nw_u, exclude)

    pattern = _random_chroma(10, seed=1)
    repeated_chroma = np.concatenate([pattern, pattern])  # 정확히 반복
    sim_r, nw_r = build_similarity_matrix(repeated_chroma, window)
    amb_r = max_sim_ambiguity(sim_r, nw_r, exclude)

    assert np.nanmean(amb_r) > np.nanmean(amb_u)
    # 정확히 반복된 시퀀스는 최댓값이 거의 1에 도달해야 함
    assert np.nanmax(amb_r) > 0.99


def test_transpose_invariant_recognizes_shifted_repeat():
    """같은 패턴을 피치클래스만 순환이동시켜 반복하면, transpose_invariant=True일
    때만 그 반복을 '닮음'으로 인식해야 한다."""
    window, exclude = 3, 4
    pattern = _random_chroma(10, seed=2)
    shifted = np.roll(pattern, shift=3, axis=1)  # 피치클래스 3칸 이동 = 조옮김
    chroma = np.concatenate([pattern, shifted])

    sim_plain, nw = build_similarity_matrix(chroma, window, transpose_invariant=False)
    amb_plain = max_sim_ambiguity(sim_plain, nw, exclude)

    sim_ti, _ = build_similarity_matrix(chroma, window, transpose_invariant=True)
    amb_ti = max_sim_ambiguity(sim_ti, nw, exclude)

    assert np.nanmean(amb_ti) > np.nanmean(amb_plain)
    assert np.nanmax(amb_ti) > 0.99


def test_cluster_entropy_exact_k_repeats_gives_k_minus_self():
    """정확히 K번 반복되는 패턴이면, 자기 자신을 뺀 다른 K-1개 복사본이 뚜렷한
    후보로 잡혀 exp(entropy) ≈ K-1이어야 한다(ENGINEERING_LOG.md v7 절 —
    '수식 검산 완료' 주장의 회귀 테스트). exclude는 진짜 자기자신만 제외할 만큼
    작게(패턴 길이보다 충분히 작게) 잡아야 이웃 복사본까지 잘못 제외되지 않는다."""
    window, exclude, merge_frames = 3, 1, 2
    pattern = _random_chroma(12, seed=3)
    k = 4
    chroma = np.concatenate([pattern] * k)

    sim, nw = build_similarity_matrix(chroma, window, transpose_invariant=False)
    amb, _ = cluster_entropy_ambiguity(sim, nw, exclude, merge_frames, temperature=0.001)

    # 각 복사본 내부, 경계에서 떨어진 안전한 프레임들만 확인 (경계 프레임은
    # 다음 복사본과 겹쳐 값이 흔들릴 수 있음)
    safe_indices = [i for i in range(nw) if i % 12 in (2, 3, 4, 5)]
    safe_vals = amb[safe_indices]
    safe_vals = safe_vals[~np.isnan(safe_vals)]
    assert safe_vals.size > 0
    assert np.allclose(safe_vals, k - 1, atol=1e-6)


def test_max_sim_with_distance_matches_maxsim_and_finds_far_repeat():
    """추가한 max_sim_with_distance가 (1) 기존 maxsim과 값이 같고
    (2) 먼 반복까지의 거리를 정확히 보고하는지."""
    import numpy as np
    from score_following.ambiguity import max_sim_ambiguity, max_sim_with_distance

    n = 60
    sim = np.full((n, n), 0.1)
    np.fill_diagonal(sim, 1.0)
    # 윈도우 5와 윈도우 45가 서로 매우 닮았다고 심어둠(거리 40 = 구조적 반복)
    sim[5, 45] = sim[45, 5] = 0.95
    # 윈도우 5 바로 옆(거리 2)에도 약한 유사가 있지만 exclude로 빠져야 함
    sim[5, 7] = sim[7, 5] = 0.99

    exclude = 10
    base = max_sim_ambiguity(sim, n, exclude)
    amb, dist = max_sim_with_distance(sim, n, exclude)
    assert np.allclose(base, amb, equal_nan=True), "기존 maxsim과 값이 달라짐"
    assert np.isclose(amb[5], 0.95), f"먼 반복(0.95)을 못 찾음: {amb[5]}"
    assert dist[5] == 40, f"거리가 40이어야 하는데 {dist[5]}"


def test_large_exclude_suppresses_nearby_loop_repeat():
    """구조 모호성 측정 방식 확인 — exclude를 키우면 가까운 루프 반복이 후보에서 빠진다."""
    import numpy as np
    from score_following.ambiguity import max_sim_with_distance

    n = 80
    sim = np.full((n, n), 0.1)
    np.fill_diagonal(sim, 1.0)
    sim[20, 24] = sim[24, 20] = 0.98   # 가까운 반복(거리 4) = 국소 모호성
    sim[20, 70] = sim[70, 20] = 0.60   # 먼 반복(거리 50) = 구조 모호성

    amb_local, dist_local = max_sim_with_distance(sim, n, exclude=2)
    amb_struct, dist_struct = max_sim_with_distance(sim, n, exclude=20)
    assert np.isclose(amb_local[20], 0.98) and dist_local[20] == 4
    assert np.isclose(amb_struct[20], 0.60) and dist_struct[20] == 50
