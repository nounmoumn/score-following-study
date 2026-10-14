# -*- coding: utf-8 -*-
"""통계 — Fisher z 메타분석 + 셔플 음성 대조. 원본: scripts/p1_verification.py main()"""
import numpy as np
from scipy.stats import norm, spearmanr


def fisher_z_meta_analysis(per_piece_rs, per_piece_ns):
    """곡별 Spearman rho 리스트를 Fisher z 변환 후 (n-3) 가중평균, 역변환.
    풀링(전체 이어붙이기) 대신 이 방법을 쓰는 이유: 곡마다 평균 난이도가 다르면
    풀링이 "곡 단위 차이"를 "국소 관계"로 착각하는 교란이 생긴다
    (ENGINEERING_LOG.md "검증 중 발견·수정한 실제 버그" #5).

    반환: (r_combined, z_stat, p_combined)"""
    rs = np.asarray(per_piece_rs, dtype=float)
    ns = np.asarray(per_piece_ns, dtype=float)
    z_vals = np.arctanh(np.clip(rs, -0.999, 0.999))
    weights = ns - 3
    z_combined = np.sum(z_vals * weights) / np.sum(weights)
    r_combined = np.tanh(z_combined)
    se = 1 / np.sqrt(np.sum(weights))
    z_stat = z_combined / se
    p_combined = 2 * (1 - norm.cdf(abs(z_stat)))
    return float(r_combined), float(z_stat), float(p_combined)


def shuffle_negative_control(per_piece_data, n_shuffles=200, seed=42):
    """각 곡 내에서 모호성을 셔플하고 같은 메타분석을 반복 — 진짜 상관이
    우연이 아님을 보이는 음성 대조.

    per_piece_data: [(errors: array, ambiguities: array), ...] 곡별 리스트.
    반환: shuffle_r_combined (array, shape (n_shuffles,))"""
    rng = np.random.default_rng(seed)
    shuffle_r_combined = []
    for _ in range(n_shuffles):
        zs = []
        for errs, ambs in per_piece_data:
            shuffled = rng.permutation(ambs)
            r, _ = spearmanr(shuffled, errs)
            n = len(errs)
            zs.append((np.arctanh(np.clip(r, -0.999, 0.999)), n - 3))
        zvals = np.array([z for z, w in zs])
        wts = np.array([w for z, w in zs])
        zc = np.sum(zvals * wts) / np.sum(wts)
        shuffle_r_combined.append(np.tanh(zc))
    return np.array(shuffle_r_combined)
