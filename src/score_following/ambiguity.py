# -*- coding: utf-8 -*-
"""
모호성 척도 — maxsim(v1~v3 계열)과 v7(클러스터링된 실질후보개수) 둘 다.

⚠️ 2026-09-07 FEEDBACK.md 전략 결정: **maxsim이 P1의 공식/보고 지표.**
v7은 폐기 아님(ASAP에서 유의, rho=+0.075 재확인됨) — 단 week4류 조건(실제
오디오+단순반복 코드진행)에서 부호가 반대로 나와 보조 지표로 강등.
자세한 근거는 FEEDBACK.md "최종 결정 (전략)" 항목, ENGINEERING_LOG.md
"모호성 척도 v1→v7" / "최댓값유사도 vs 실질후보개수A 불일치" 절 참조.

⚠️ 계산식 자체는 "재설계 금지"로 확정됨(RESEARCH_DEFINITION.md). 이 모듈은
scripts/exploratory/{p1_verification,ambiguity_v5,ambiguity_v7}.py의 로직을
그대로 옮긴 것이며 수식을 바꾸지 않았다.

원본:
  - build_similarity_matrix(transpose_invariant=False): p1_verification.frame_ambiguity의
    feats 구성 로직 (원조 ASAP 헤드라인 rho=+0.196의 근거)
  - build_similarity_matrix(transpose_invariant=True): ambiguity_v5.build_transpose_sim_matrix
    (v7 및 2026-09-07 재검증에서 쓴 조옮김 불변 버전, rho=+0.173)
  - max_sim_ambiguity: 위 두 원본 공통 로직(자기 주변 제외 최댓값)
  - cluster_entropy_ambiguity: ambiguity_v7.py (exp(entropy) = 실질후보개수, 피크 클러스터링)
"""
import numpy as np

PITCH_CLASSES = 12

DEFAULT_PEAK_REL_THRESHOLD = 0.01


def build_similarity_matrix(chroma, window, transpose_invariant=False):
    """chroma: (n_frames, 12). window프레임을 이어붙인 문맥벡터들의 코사인 유사도 행렬.

    transpose_invariant=True면 12개 조옮김(피치클래스 순환이동) 중 최댓값을 취한다
    (Müller & Clausen 2007 방식) — v7 및 2026-09-07 재검증에 쓰인 버전.
    transpose_invariant=False면 원본 p1_verification.py의 방식(조옮김 없음) —
    ASAP 헤드라인 결과(rho=+0.196, z=71.98)의 근거.
    """
    n = len(chroma)
    n_windows = n - window
    if n_windows < 1:
        return np.zeros((0, 0)), 0

    if not transpose_invariant:
        feats = np.zeros((n_windows, window * PITCH_CLASSES))
        for i in range(n_windows):
            feats[i] = chroma[i:i + window].reshape(-1)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        sim = feats @ feats.T
        return sim, n_windows

    tensor = np.stack([chroma[i:i + window] for i in range(n_windows)])

    def flatten_norm(t):
        f = t.reshape(n_windows, -1)
        nrm = np.linalg.norm(f, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return f / nrm

    feats0 = flatten_norm(tensor)
    best_sim = feats0 @ feats0.T
    for k in range(1, PITCH_CLASSES):
        rotated = np.roll(tensor, shift=k, axis=2)
        feats_k = flatten_norm(rotated)
        sim_k = feats0 @ feats_k.T
        best_sim = np.maximum(best_sim, sim_k)
    return best_sim, n_windows


def max_sim_ambiguity(sim, n_windows, exclude):
    """각 윈도우에서, 자기 주변(±exclude)을 제외한 나머지와의 최댓값 유사도.
    "가장 닮은 곳 딱 하나"의 유사도 — v1~v3 계열, 지금은 P1 공식 지표."""
    out = np.full(n_windows, np.nan)
    for i in range(n_windows):
        lo, hi = max(0, i - exclude), min(n_windows, i + exclude + 1)
        mask = np.ones(n_windows, dtype=bool)
        mask[lo:hi] = False
        if mask.any():
            out[i] = sim[i, mask].max()
    return out


def max_sim_with_distance(sim, n_windows, exclude):
    """`max_sim_ambiguity`와 같은 값 + **가장 닮은 곳이 몇 윈도우 떨어져 있는지**를 함께 반환.

    왜 필요한가(2026-09-17 추가): RESEARCH_DEFINITION.md §2는 모호성을 두 종류로 나눈다 —
    **국소(화성)**(4마디 루프 안에서 이 C가 몇 번째 C인지, 경쟁 후보가 몇 초 거리)와
    **구조(형식)**(1절 vs 2절, 경쟁 후보가 수십 초 거리). 둘을 가르는 정의적 특징이 바로
    **경쟁 후보까지의 거리**이므로, 그 거리를 기록해두면 어느 종류의 모호성인지 사후에
    분류할 수 있다.

    ⚠️ 계산식은 `max_sim_ambiguity`와 **완전히 동일**하다(재설계 금지 준수). 부가 정보만 더함.
    구조 모호성 자체는 이 함수가 아니라 **exclude를 크게 주는 것**으로 측정한다 —
    exclude가 크면 가까운 루프 반복이 후보에서 빠지고 먼 구조적 반복만 남는다.

    반환: (ambiguity, distance_windows) — 둘 다 shape (n_windows,), 후보가 없으면 nan.
    """
    out = np.full(n_windows, np.nan)
    dist = np.full(n_windows, np.nan)
    for i in range(n_windows):
        lo, hi = max(0, i - exclude), min(n_windows, i + exclude + 1)
        mask = np.ones(n_windows, dtype=bool)
        mask[lo:hi] = False
        if not mask.any():
            continue
        idxs = np.nonzero(mask)[0]
        k = int(np.argmax(sim[i, idxs]))
        out[i] = sim[i, idxs[k]]
        dist[i] = abs(int(idxs[k]) - i)
    return out, dist


def cluster_entropy_ambiguity(sim, n_windows, exclude, merge_frames, temperature,
                               peak_rel_threshold=DEFAULT_PEAK_REL_THRESHOLD):
    """v7: exp(Shannon 엔트로피) = 실질적 후보 개수.

    각 행에서: softmax로 확률화 -> '피크(유사도가 실제로 높은 후보)'만 골라
    그것들끼리 가까우면 클러스터로 병합 -> 배경은 그대로 -> 엔트로피 -> exp.
    (배경까지 무조건 병합하면 곡 전체가 1~2 덩어리로 뭉개지는 버그가 있었음 —
    2026-09-06에 발견·수정됨. 피크만 골라내는 현재 방식이 그 수정.)

    반환: (ambiguity, n_clusters) — 둘 다 shape (n_windows,)
    """
    ambiguity = np.full(n_windows, np.nan)
    n_clusters_arr = np.full(n_windows, np.nan)
    for i in range(n_windows):
        lo, hi = max(0, i - exclude), min(n_windows, i + exclude + 1)
        mask = np.ones(n_windows, dtype=bool)
        mask[lo:hi] = False
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            continue
        logits = sim[i, idxs] / temperature
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()

        peak_mask = p >= (p.max() * peak_rel_threshold)
        peak_idxs = idxs[peak_mask]
        peak_p = p[peak_mask]
        background_p = p[~peak_mask]

        cluster_probs = []
        if len(peak_idxs) > 0:
            order = np.argsort(peak_idxs)
            peak_idxs_sorted = peak_idxs[order]
            peak_p_sorted = peak_p[order]
            cur_sum = peak_p_sorted[0]
            for k in range(1, len(peak_idxs_sorted)):
                gap = peak_idxs_sorted[k] - peak_idxs_sorted[k - 1]
                if gap <= merge_frames:
                    cur_sum += peak_p_sorted[k]
                else:
                    cluster_probs.append(cur_sum)
                    cur_sum = peak_p_sorted[k]
            cluster_probs.append(cur_sum)

        all_probs = np.array(cluster_probs + list(background_p))
        all_probs = all_probs[all_probs > 1e-12]
        if len(all_probs) == 0:
            continue

        ent = -np.sum(all_probs * np.log(all_probs))
        ambiguity[i] = np.exp(ent)
        n_clusters_arr[i] = len(cluster_probs)
    return ambiguity, n_clusters_arr
