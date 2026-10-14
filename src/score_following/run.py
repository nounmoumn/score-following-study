# -*- coding: utf-8 -*-
"""
config 기반 실험 실행기. 사용법:

    python -m score_following.run --config configs/paper/p1_asap_maxsim.yaml

각 실행은 results/<experiment>_<timestamp>/ 아래에 config.yaml(사본),
metrics.json, frame_results.csv(week4) 또는 piece_results.csv(asap),
run.log, environment.txt를 남긴다.

지원하는 kind:
  - p1_asap:      ASAP 데이터셋으로 P1(모호성→오차) 검증. 원본: scripts/p1_verification.py
  - p1_week4:     week4 실제 기타 녹음으로 P1 검증. 원본: week4_p1_test_v2.py, dlba_p1_test.py
  - p2_asap_gap:  ASAP으로 P2(온라인 vs 오프라인 격차) 검증. 2026-09-08 신규,
                  FEEDBACK.md "P2 설계" 절 · offline_dtw.py 참조.
  - p2_week4_gap: week4로 P2 외적 타당성 검증(open_end=True). 2026-09-08 신규.
  - p3_asap_order: ASAP으로 P3(모호성→드리프트 시간순서) 검증. 2026-09-17 신규,
                  drift.py 참조.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from ambiguity import build_similarity_matrix, max_sim_ambiguity, cluster_entropy_ambiguity
from drift import (before_after_window_test, circular_shift_null, cross_correlation_by_lag,
                    drift_rate, largest_drift_episode, peak_lag, smooth, upward_crossings,
                    window_mean_percentile)
from features import load_notes, notes_to_chroma_stream, audio_chroma, build_score_chroma_v2
from ground_truth import piecewise_linear_gt
from offline_dtw import OfflineDTW
from online_dtw import OnlineDTW
from stats import fisher_z_meta_analysis, shuffle_negative_control

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


_LOCAL_PATHS_CACHE = None


def _local_paths():
    """data/local_paths.yaml(개인 환경 전용, .gitignore됨)에서 ${변수} 치환값을 읽는다.
    없으면 빈 dict — 그러면 ${...}가 남아있는 경로는 그대로 실패하며, 그 자체가
    "로컬 경로 설정 안 했다"는 명확한 에러가 된다. data/README.md 참조."""
    global _LOCAL_PATHS_CACHE
    if _LOCAL_PATHS_CACHE is None:
        p = os.path.join(REPO_ROOT, "data", "local_paths.yaml")
        _LOCAL_PATHS_CACHE = {}
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                _LOCAL_PATHS_CACHE = yaml.safe_load(f) or {}
    return _LOCAL_PATHS_CACHE


def _resolve(path):
    """설정 파일의 경로를 해석한다.
    1. ${key} 형태는 data/local_paths.yaml의 key 값으로 치환 (개인 절대경로, 코드에 안 박음)
    2. 그 결과가 절대경로면 그대로, 상대경로면 저장소 루트 기준으로 해석."""
    if path is None:
        return None
    for key, val in _local_paths().items():
        path = path.replace("${" + key + "}", str(val))
    if "${" in path:
        raise ValueError(
            f"경로에 치환 안 된 ${{...}}가 남아있음: {path!r}. "
            f"data/local_paths.yaml에 해당 키를 추가할 것 (data/local_paths.example.yaml 참조)."
        )
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _diverse_rows(rows):
    """작곡가별 라운드로빈 — 같은 곡 반복측정 대신 다양성 확보. 원본: p1_verification.py"""
    by_composer = defaultdict(list)
    for row in rows:
        by_composer[row["composer"]].append(row)
    out = []
    lists = list(by_composer.values())
    idx = 0
    while any(idx < len(lst) for lst in lists):
        for lst in lists:
            if idx < len(lst):
                out.append(lst[idx])
        idx += 1
    return out


def _piece_ambiguity_and_error(perf_notes, score_notes, cfg, band_width_frames):
    """ASAP 한 곡: 크로마 생성 -> 정렬 -> 모호성/오차 프레임 리스트 반환."""
    hop_U = cfg["feature"]["frame_hop_sec"]
    perf_dur = max(n[1] for n in perf_notes)
    score_dur = max(n[1] for n in score_notes)
    tempo_ratio = score_dur / perf_dur
    hop_V = hop_U * tempo_ratio

    perf_chroma = notes_to_chroma_stream(perf_notes, perf_dur, hop=hop_U)
    score_chroma = notes_to_chroma_stream(score_notes, score_dur, hop=hop_V)
    if len(perf_chroma) < 50 or len(score_chroma) < 50:
        return None

    window = max(2, int(round(cfg["feature"]["ambiguity_window_sec"] / hop_V)))
    exclude = max(1, int(round(cfg["feature"]["ambiguity_exclude_sec"] / hop_V)))
    sim, n_windows = build_similarity_matrix(
        score_chroma, window, transpose_invariant=cfg["feature"]["transpose_invariant"]
    )
    if n_windows < exclude * 2 + 2:
        return None

    metric = cfg["ambiguity_metric"]
    if metric == "maxsim":
        amb_full = max_sim_ambiguity(sim, n_windows, exclude)
    elif metric == "v7":
        v7cfg = cfg.get("v7", {})
        amb_full, _ = cluster_entropy_ambiguity(
            sim, n_windows, exclude,
            merge_frames=max(1, int(round(v7cfg.get("merge_sec", 2.0) / hop_V))),
            temperature=v7cfg.get("temperature", 0.001),
            peak_rel_threshold=v7cfg.get("peak_rel_threshold", 0.01),
        )
    else:
        raise ValueError(f"unknown ambiguity_metric: {metric}")

    return dict(hop_U=hop_U, hop_V=hop_V, perf_chroma=perf_chroma, score_chroma=score_chroma,
                n_windows=n_windows, exclude=exclude, amb_full=amb_full)


def run_p1_asap(cfg, out_dir, log):
    import pretty_midi  # noqa: F401  (실패 시 조기 에러가 낫다)

    asap_root = _resolve(cfg["data"]["asap_root"])
    annot_path = os.path.join(asap_root, "asap_annotations.json")
    with open(annot_path, encoding="utf-8") as f:
        annot = json.load(f)
    with open(os.path.join(asap_root, "metadata.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if cfg["data"].get("diverse_sampling", True):
        rows = _diverse_rows(rows)

    band_width_frames = int(round(cfg["tracker"]["band_width_sec"] / cfg["feature"]["frame_hop_sec"]))
    health_threshold = cfg["evaluation"]["health_threshold_sec"]
    n_pieces = cfg["data"]["n_pieces"]

    all_data = []  # [(label, [(perf_time, error, amb), ...]), ...] 건강한 곡만
    unhealthy = []
    tested = 0
    frame_rows_out = []  # frame_results.csv용

    for row in rows:
        if tested >= n_pieces:
            break
        label = f"{row['composer']}/{row['title']}"
        ann = annot.get(row["midi_performance"])
        if ann is None or not ann.get("score_and_performance_aligned", False):
            continue
        perf_path = os.path.join(asap_root, row["midi_performance"])
        score_path = os.path.join(asap_root, row["midi_score"])
        if not (os.path.exists(perf_path) and os.path.exists(score_path)):
            continue
        try:
            perf_notes = load_notes(perf_path)
            score_notes = load_notes(score_path)
            if not perf_notes or not score_notes:
                continue
            prep = _piece_ambiguity_and_error(perf_notes, score_notes, cfg, band_width_frames)
            if prep is None:
                continue
            gt_fn, gt_lo, gt_hi = piecewise_linear_gt(
                ann["performance_beats"], ann["midi_score_beats"]
            )
            dtw = OnlineDTW(prep["perf_chroma"], prep["score_chroma"], band_width=band_width_frames)
            path = dtw.run()
            est_j_of_t = {}
            for (t, j) in path:
                est_j_of_t[t] = j

            result = []
            for t, j in sorted(est_j_of_t.items()):
                perf_time = t * prep["hop_U"]
                if not (gt_lo <= perf_time <= gt_hi):
                    continue
                true_score_time = float(gt_fn(perf_time))
                est_score_time = j * prep["hop_V"]
                error = abs(est_score_time - true_score_time)
                true_j = int(round(true_score_time / prep["hop_V"]))
                if 0 <= true_j < prep["n_windows"] and not np.isnan(prep["amb_full"][true_j]):
                    result.append((perf_time, error, prep["amb_full"][true_j]))
        except Exception as e:
            log(f"  {label}: 실패 ({e})")
            continue

        if len(result) < 20:
            continue
        tested += 1
        errors = np.array([r[1] for r in result])
        med = float(np.median(errors))
        log(f"[{tested}/{n_pieces}] {label}  n={len(result)}  median_error={med:.3f}s")
        for perf_time, error, amb in result:
            frame_rows_out.append((label, perf_time, error, amb))
        if med > health_threshold:
            unhealthy.append(label)
            continue
        all_data.append((label, result))

    if not all_data:
        raise RuntimeError("건강한 곡이 하나도 없음 — 설정을 확인할 것")

    per_piece_rs, per_piece_ns, piece_rows_out, shuffle_input = [], [], [], []
    for label, data in all_data:
        errs = np.array([r[1] for r in data])
        ambs = np.array([r[2] for r in data])
        r, p = spearmanr(ambs, errs)
        per_piece_rs.append(r)
        per_piece_ns.append(len(data))
        piece_rows_out.append((label, len(data), r, p))
        shuffle_input.append((errs, ambs))

    r_combined, z_stat, p_combined = fisher_z_meta_analysis(per_piece_rs, per_piece_ns)
    n_shuffles = cfg["evaluation"].get("n_shuffles", 200)
    shuffle_dist = shuffle_negative_control(shuffle_input, n_shuffles=n_shuffles,
                                             seed=cfg.get("seed", 42))
    shuffle_z = (r_combined - shuffle_dist.mean()) / (shuffle_dist.std() + 1e-9)

    log(f"\n결합 rho={r_combined:+.4f}  z={z_stat:.2f}  p={p_combined:.2e}")
    log(f"건강한 곡 {len(all_data)}개 / 탈락 {len(unhealthy)}개")
    log(f"음성대조 대비 이탈: z={shuffle_z:.2f}")

    with open(os.path.join(out_dir, "piece_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_frames", "rho", "p_value"])
        w.writerows(piece_rows_out)

    with open(os.path.join(out_dir, "frame_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "perf_time_sec", "error_sec", "ambiguity"])
        w.writerows(frame_rows_out)

    return {
        "rho": r_combined, "z": z_stat, "p_value": p_combined,
        "n_healthy_pieces": len(all_data), "n_unhealthy_pieces": len(unhealthy),
        "unhealthy_pieces": unhealthy,
        "shuffle_control_z": float(shuffle_z),
        "n_frames_total": len(frame_rows_out),
    }


def run_p2_asap_gap(cfg, out_dir, log):
    """P2: 온라인(band 제약, 인과적) vs 오프라인(전역, 제약 없음) 추적 오차 격차(GAP).
    설계: FEEDBACK.md "P2 설계 (온라인 vs 오프라인 격차)" (전략 트랙, 2026-09-08).

    GAP(t) = online_err(t) - offline_err(t). 기대: 모호한 구간일수록 GAP이 커짐
    (오프라인은 전역 정보로 버티는데 온라인은 못 버팀).

    ⚠️ 건강도 필터를 GAP 상관에도 그대로 적용한다(온라인 오차 중앙값 기준, P1과 동일
    임계값·이유). 근거(스모크 테스트, ENGINEERING_LOG.md "P2 오프라인 global DTW 구현"
    절): 온라인이 파국적으로 완전히 길을 잃은 곡(Liszt/Ballade_2)에서 rho(모호성,GAP)가
    **반대 부호**로 나옴 — 길 잃은 뒤의 오차는 드리프트 누적이 지배해서 그 순간의 국소
    모호성과 무관해진다는, P1에서 이미 확인된 것과 같은 오염 메커니즘으로 추정됨."""
    import pretty_midi  # noqa: F401

    asap_root = _resolve(cfg["data"]["asap_root"])
    annot_path = os.path.join(asap_root, "asap_annotations.json")
    with open(annot_path, encoding="utf-8") as f:
        annot = json.load(f)
    with open(os.path.join(asap_root, "metadata.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if cfg["data"].get("diverse_sampling", True):
        rows = _diverse_rows(rows)

    band_width_frames = int(round(cfg["tracker"]["band_width_sec"] / cfg["feature"]["frame_hop_sec"]))
    health_threshold = cfg["evaluation"]["health_threshold_sec"]
    n_pieces = cfg["data"]["n_pieces"]

    all_data = []  # [(label, [(perf_time, gap, amb), ...]), ...] 건강한 곡만(온라인 기준)
    unhealthy = []
    tested = 0
    frame_rows_out = []  # frame_results.csv용: label, perf_time, online_err, offline_err, gap, amb
    piece_diag_rows = []  # 곡별 진단: online/offline median error, offline 런타임

    for row in rows:
        if tested >= n_pieces:
            break
        label = f"{row['composer']}/{row['title']}"
        ann = annot.get(row["midi_performance"])
        if ann is None or not ann.get("score_and_performance_aligned", False):
            continue
        perf_path = os.path.join(asap_root, row["midi_performance"])
        score_path = os.path.join(asap_root, row["midi_score"])
        if not (os.path.exists(perf_path) and os.path.exists(score_path)):
            continue
        try:
            perf_notes = load_notes(perf_path)
            score_notes = load_notes(score_path)
            if not perf_notes or not score_notes:
                continue
            prep = _piece_ambiguity_and_error(perf_notes, score_notes, cfg, band_width_frames)
            if prep is None:
                continue
            gt_fn, gt_lo, gt_hi = piecewise_linear_gt(
                ann["performance_beats"], ann["midi_score_beats"]
            )

            online = OnlineDTW(prep["perf_chroma"], prep["score_chroma"], band_width=band_width_frames)
            online.run()
            online_j_of_t = {}
            for (t, j) in online.path:
                online_j_of_t[t] = j

            t0 = time.time()
            offline = OfflineDTW(prep["perf_chroma"], prep["score_chroma"], open_end=False)
            offline.run()
            offline_runtime = time.time() - t0
            offline_j_of_t = offline.j_of_t()

            result = []
            piece_online_errs, piece_offline_errs = [], []
            for t in range(len(prep["perf_chroma"])):
                if t not in online_j_of_t or t not in offline_j_of_t:
                    continue
                perf_time = t * prep["hop_U"]
                if not (gt_lo <= perf_time <= gt_hi):
                    continue
                true_score_time = float(gt_fn(perf_time))
                online_err = abs(online_j_of_t[t] * prep["hop_V"] - true_score_time)
                offline_err = abs(offline_j_of_t[t] * prep["hop_V"] - true_score_time)
                true_j = int(round(true_score_time / prep["hop_V"]))
                if not (0 <= true_j < prep["n_windows"]) or np.isnan(prep["amb_full"][true_j]):
                    continue
                gap = online_err - offline_err
                result.append((perf_time, gap, prep["amb_full"][true_j], online_err, offline_err))
                piece_online_errs.append(online_err)
                piece_offline_errs.append(offline_err)
                frame_rows_out.append((label, perf_time, online_err, offline_err, gap,
                                        prep["amb_full"][true_j]))
        except Exception as e:
            log(f"  {label}: 실패 ({e})")
            continue

        if len(result) < 20:
            continue
        tested += 1
        online_med = float(np.median(piece_online_errs))
        offline_med = float(np.median(piece_offline_errs))
        log(f"[{tested}/{n_pieces}] {label}  n={len(result)}  "
            f"online_med={online_med:.3f}s  offline_med={offline_med:.3f}s  "
            f"offline_runtime={offline_runtime:.1f}s")
        piece_diag_rows.append((label, len(result), online_med, offline_med, offline_runtime))
        if online_med > health_threshold:
            log(f"  ⚠️ 건강도 필터 탈락 (온라인 중앙값 {online_med:.3f}s > {health_threshold}s)")
            unhealthy.append(label)
            continue
        all_data.append((label, result))

    if not all_data:
        raise RuntimeError("건강한 곡이 하나도 없음 — 설정을 확인할 것")

    per_piece_rs, per_piece_ns, piece_rows_out, shuffle_input = [], [], [], []
    for label, data in all_data:
        gaps = np.array([r[1] for r in data])
        ambs = np.array([r[2] for r in data])
        r, p = spearmanr(ambs, gaps)
        per_piece_rs.append(r)
        per_piece_ns.append(len(data))
        piece_rows_out.append((label, len(data), r, p))
        shuffle_input.append((gaps, ambs))

    r_combined, z_stat, p_combined = fisher_z_meta_analysis(per_piece_rs, per_piece_ns)
    n_shuffles = cfg["evaluation"].get("n_shuffles", 200)
    shuffle_dist = shuffle_negative_control(shuffle_input, n_shuffles=n_shuffles,
                                             seed=cfg.get("seed", 42))
    shuffle_z = (r_combined - shuffle_dist.mean()) / (shuffle_dist.std() + 1e-9)

    log(f"\n결합 rho(모호성,GAP)={r_combined:+.4f}  z={z_stat:.2f}  p={p_combined:.2e}")
    log(f"건강한 곡(온라인 기준) {len(all_data)}개 / 탈락 {len(unhealthy)}개")
    log(f"음성대조 대비 이탈: z={shuffle_z:.2f}")

    with open(os.path.join(out_dir, "piece_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_frames", "rho", "p_value"])
        w.writerows(piece_rows_out)

    with open(os.path.join(out_dir, "piece_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_frames", "online_median_error_sec", "offline_median_error_sec",
                     "offline_runtime_sec"])
        w.writerows(piece_diag_rows)

    with open(os.path.join(out_dir, "frame_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "perf_time_sec", "online_error_sec", "offline_error_sec", "gap_sec",
                     "ambiguity"])
        w.writerows(frame_rows_out)

    # 부가 진단: rho(모호성, online_err), rho(모호성, offline_err) 각각도 같은 건강한 곡
    # 집합·같은 Fisher z 절차로 — "GAP이 양수인 게 offline은 안 그런데 online만 그래서"
    # 인지 직접 확인하기 위함(ENGINEERING_LOG.md "다음 확인 필요" 1번).
    extra = {}
    for extra_name, idx in (("online_err", 3), ("offline_err", 4)):
        rs, ns = [], []
        for label, data in all_data:
            errs = np.array([r[idx] for r in data])
            ambs = np.array([r[2] for r in data])
            r, _ = spearmanr(ambs, errs)
            rs.append(r)
            ns.append(len(data))
        r_c, z_c, p_c = fisher_z_meta_analysis(rs, ns)
        extra[f"rho_{extra_name}"] = r_c
        extra[f"z_{extra_name}"] = z_c
        extra[f"p_{extra_name}"] = p_c
        log(f"rho(모호성,{extra_name})={r_c:+.4f}  z={z_c:.2f}  p={p_c:.2e}")

    return {
        "rho": r_combined, "z": z_stat, "p_value": p_combined,
        "n_healthy_pieces": len(all_data), "n_unhealthy_pieces": len(unhealthy),
        "unhealthy_pieces": unhealthy,
        "shuffle_control_z": float(shuffle_z),
        "n_frames_total": len(frame_rows_out),
        **extra,
    }


def _longest_contiguous(valid_mask):
    """True가 연속으로 가장 길게 이어진 구간의 (시작, 끝+1)을 반환.
    교차상관은 **등간격 연속 시계열**이어야 하므로, P1/P2처럼 유효 프레임만
    골라내서 띄어놓은 배열을 그대로 쓰면 안 된다(간격이 불규칙해지면 lag의
    의미가 깨진다). 그래서 가장 긴 연속 구간만 잘라 쓴다."""
    best_len, best = 0, (0, 0)
    i = 0
    n = len(valid_mask)
    while i < n:
        if not valid_mask[i]:
            i += 1
            continue
        j = i
        while j < n and valid_mask[j]:
            j += 1
        if j - i > best_len:
            best_len, best = j - i, (i, j)
        i = j
    return best


def run_p3_asap_order(cfg, out_dir, log):
    """P3: "모호성 진입 → (위치 상실) → 드리프트 시작"의 시간 순서 검정.

    핵심 통계: `amb(t)`와 `drift_rate(t)`의 교차상관 피크가 **양의 lag**에 있는가
    (= 모호성이 드리프트보다 먼저 오는가). 귀무가설은 순환이동(자기상관 보존).
    설계 근거와 전략 원안에서 바꾼 점 2가지는 `drift.py` 모듈 docstring 참조.

    ⚠️ 건강도 필터를 **기본적으로 끈다**(`health_threshold_sec: null`). P1/P2와 정반대인데
    이유가 있다: P3가 보려는 사건이 바로 '드리프트'이므로, 한 번도 안 드리프트한 곡은
    관측할 사건이 없어서 정보가 없고, 파국적으로 실패한 곡이 오히려 가장 선명한 사건을
    준다. 대신 곡별로 순환이동 귀무분포와 비교하므로 "그 곡이 원래 많이 틀린다"는
    곡 단위 차이는 상쇄된다."""
    import pretty_midi  # noqa: F401

    asap_root = _resolve(cfg["data"]["asap_root"])
    with open(os.path.join(asap_root, "asap_annotations.json"), encoding="utf-8") as f:
        annot = json.load(f)
    with open(os.path.join(asap_root, "metadata.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if cfg["data"].get("diverse_sampling", True):
        rows = _diverse_rows(rows)

    hop = cfg["feature"]["frame_hop_sec"]
    band_width_frames = int(round(cfg["tracker"]["band_width_sec"] / hop))
    n_pieces = cfg["data"]["n_pieces"]
    p3 = cfg["p3"]
    max_lag_frames = int(round(p3["max_lag_sec"] / hop))
    smooth_frames = int(round(p3["smooth_sec"] / hop))
    window_frames = int(round(p3["event_window_sec"] / hop))
    min_gap_frames = int(round(p3.get("event_min_gap_sec", 2.0) / hop))
    amb_pct = p3.get("ambiguity_percentile", 75)
    drift_pct = p3.get("drift_percentile", 75)
    health = cfg["evaluation"].get("health_threshold_sec", None)
    n_shuffles = cfg["evaluation"].get("n_shuffles", 200)

    piece_rows, tested = [], 0
    tot_before = tot_after = tot_trigger = 0

    for row in rows:
        if tested >= n_pieces:
            break
        label = f"{row['composer']}/{row['title']}"
        ann = annot.get(row["midi_performance"])
        if ann is None or not ann.get("score_and_performance_aligned", False):
            continue
        perf_path = os.path.join(asap_root, row["midi_performance"])
        score_path = os.path.join(asap_root, row["midi_score"])
        if not (os.path.exists(perf_path) and os.path.exists(score_path)):
            continue
        try:
            perf_notes, score_notes = load_notes(perf_path), load_notes(score_path)
            if not perf_notes or not score_notes:
                continue
            prep = _piece_ambiguity_and_error(perf_notes, score_notes, cfg, band_width_frames)
            if prep is None:
                continue
            gt_fn, gt_lo, gt_hi = piecewise_linear_gt(
                ann["performance_beats"], ann["midi_score_beats"])

            dtw = OnlineDTW(prep["perf_chroma"], prep["score_chroma"],
                             band_width=band_width_frames)
            dtw.run()
            est_j_of_t = {}
            for (t, j) in dtw.path:
                est_j_of_t[t] = j

            m = len(prep["perf_chroma"])
            err_arr = np.full(m, np.nan)
            amb_arr = np.full(m, np.nan)
            for t in range(m):
                if t not in est_j_of_t:
                    continue
                perf_time = t * prep["hop_U"]
                if not (gt_lo <= perf_time <= gt_hi):
                    continue
                true_score_time = float(gt_fn(perf_time))
                true_j = int(round(true_score_time / prep["hop_V"]))
                if not (0 <= true_j < prep["n_windows"]):
                    continue
                if np.isnan(prep["amb_full"][true_j]):
                    continue
                err_arr[t] = abs(est_j_of_t[t] * prep["hop_V"] - true_score_time)
                amb_arr[t] = prep["amb_full"][true_j]

            lo, hi = _longest_contiguous(~np.isnan(err_arr) & ~np.isnan(amb_arr))
            if hi - lo < max(100, 4 * max_lag_frames):
                continue
            err = err_arr[lo:hi]
            amb = amb_arr[lo:hi]
        except Exception as e:
            log(f"  {label}: 실패 ({e})")
            continue

        tested += 1
        err_med = float(np.median(err))
        if health is not None and err_med > health:
            log(f"[{tested}/{n_pieces}] {label}  건강도 필터 탈락(중앙값 {err_med:.3f}s)")
            continue

        dr = drift_rate(err, hop, smooth_frames, rectify=True)
        dr_raw = drift_rate(err, hop, smooth_frames, rectify=False)
        amb_s = smooth(amb, smooth_frames)

        lags, corrs = cross_correlation_by_lag(amb_s, dr, max_lag_frames)
        pl, pc = peak_lag(lags, corrs)
        lags_r, corrs_r = cross_correlation_by_lag(amb_s, dr_raw, max_lag_frames)
        pl_raw, pc_raw = peak_lag(lags_r, corrs_r)

        null_lags, null_corrs = circular_shift_null(
            amb_s, dr, max_lag_frames, n_shuffles=n_shuffles, seed=cfg.get("seed", 42))
        null_pct = float(np.mean(null_corrs >= pc) * 100) if not np.isnan(pc) else np.nan

        amb_events = upward_crossings(amb_s, np.percentile(amb_s, amb_pct), min_gap_frames)
        dr_thresh = np.percentile(dr[dr > 0], drift_pct) if np.any(dr > 0) else np.inf
        drift_events = upward_crossings(dr, dr_thresh, min_gap_frames)
        nb, na, nt = before_after_window_test(amb_events, drift_events, window_frames, len(dr))
        tot_before += nb
        tot_after += na
        tot_trigger += nt

        # ⭐ 주 분석: 그 곡에서 가장 큰 드리프트 사건 하나 — 임계값이 필요 없다
        # (임계값 방식은 추적이 잘 된 곡에서 수십 ms 잔떨림을 사건으로 잡는 문제가 있었다,
        #  실측: Bach 오차중앙값 31ms인데 '드리프트 사건' 16개 — drift.py 주석 참조)
        ep_start, ep_end, ep_gain = largest_drift_episode(err)
        pct_before = window_mean_percentile(amb, ep_start, window_frames, side="before")
        pct_after = window_mean_percentile(amb, ep_end, window_frames, side="after")

        log(f"[{tested}/{n_pieces}] {label}  n={hi-lo}  오차중앙값={err_med:.3f}s  "
            f"최대드리프트={ep_gain:.2f}s@{(ep_start or 0)*hop:.1f}s "
            f"모호성백분위(직전)={pct_before:.0f}  "
            f"피크lag={pl*hop:+.2f}s(r={pc:+.3f}, 귀무상위{null_pct:.1f}%)  "
            f"사건 before/after={nb}/{na}")
        piece_rows.append((label, hi - lo, err_med, pl * hop, pc, null_pct,
                            pl_raw * hop, pc_raw, nb, na, nt,
                            ep_gain, (ep_start or 0) * hop, pct_before, pct_after))

    if not piece_rows:
        raise RuntimeError("분석 가능한 곡이 없음 — 설정을 확인할 것")

    peak_lags_sec = np.array([r[3] for r in piece_rows], dtype=float)
    peak_corrs = np.array([r[4] for r in piece_rows], dtype=float)
    null_pcts = np.array([r[5] for r in piece_rows], dtype=float)
    raw_lags_sec = np.array([r[6] for r in piece_rows], dtype=float)
    ep_gains = np.array([r[11] for r in piece_rows], dtype=float)
    pct_befores = np.array([r[13] for r in piece_rows], dtype=float)
    pct_afters = np.array([r[14] for r in piece_rows], dtype=float)

    n_pos = int(np.sum(peak_lags_sec > 0))
    n_neg = int(np.sum(peak_lags_sec < 0))
    n_zero = int(np.sum(peak_lags_sec == 0))
    # 부호검정(양/음만, 0은 제외) — "피크가 양의 lag" 비율이 우연인지
    from scipy.stats import binomtest
    sign_p = binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else np.nan
    ba_p = (binomtest(tot_before, tot_before + tot_after, 0.5).pvalue
            if (tot_before + tot_after) > 0 else np.nan)

    # ⭐ 주 검정: 최대 드리프트 사건 직전 창의 모호성 백분위가 50보다 큰가(곡 단위 부호검정).
    # 귀무가설 아래에서 이 백분위는 0~100 균등분포이므로 중앙값 50이 기준선이고,
    # 50보다 유의하게 크면 "드리프트가 시작되기 전에 모호성이 이미 높았다"는 뜻.
    valid_ep = ~np.isnan(pct_befores)
    n_above = int(np.sum(pct_befores[valid_ep] > 50))
    n_below = int(np.sum(pct_befores[valid_ep] < 50))
    ep_p = (binomtest(n_above, n_above + n_below, 0.5).pvalue
            if (n_above + n_below) > 0 else np.nan)
    try:
        from scipy.stats import wilcoxon
        ep_wilcoxon_p = float(wilcoxon(pct_befores[valid_ep] - 50).pvalue)
    except Exception:
        ep_wilcoxon_p = float("nan")

    log("\n" + "=" * 70)
    log("⭐ 주 검정 — 최대 드리프트 사건 '직전' 창의 모호성 백분위 (귀무 = 50, 균등분포)")
    log(f"  곡 {int(valid_ep.sum())}개: 백분위 중앙값 = {np.nanmedian(pct_befores):.1f} "
        f"(50보다 크면 모호성이 드리프트보다 선행)")
    log(f"  50 초과 {n_above}곡 / 미만 {n_below}곡 — 부호검정 p={ep_p:.3g}, "
        f"Wilcoxon p={ep_wilcoxon_p:.3g}")
    log(f"  대조(사건 '직후' 창) 백분위 중앙값 = {np.nanmedian(pct_afters):.1f}")
    log(f"  최대 드리프트 상승량 중앙값 = {np.nanmedian(ep_gains):.2f}s")
    log("")
    log(f"곡 수 {len(piece_rows)}개")
    log(f"피크 lag 부호: 양(모호성 먼저) {n_pos} / 음(드리프트 먼저) {n_neg} / 0 {n_zero}"
        f"   부호검정 p={sign_p:.3g}")
    log(f"피크 lag 중앙값 = {np.nanmedian(peak_lags_sec):+.2f}s "
        f"(정류 안 한 버전 중앙값 {np.nanmedian(raw_lags_sec):+.2f}s)")
    log(f"피크 상관 중앙값 = {np.nanmedian(peak_corrs):+.3f}, "
        f"순환이동 귀무분포 상위 백분위 중앙값 = {np.nanmedian(null_pcts):.1f}% "
        f"(작을수록 우연 아님)")
    log(f"사건 기반 대칭창 검정: 드리프트 시작 {tot_trigger}개 중 "
        f"직전 {cfg['p3']['event_window_sec']}초에 모호진입 있었음 {tot_before}개 / "
        f"직후 {tot_after}개   이항검정 p={ba_p:.3g}")

    with open(os.path.join(out_dir, "piece_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_frames", "median_error_sec", "peak_lag_sec", "peak_corr",
                     "null_percentile", "peak_lag_sec_unrectified", "peak_corr_unrectified",
                     "n_before", "n_after", "n_drift_events",
                     "max_drift_gain_sec", "max_drift_onset_sec",
                     "amb_pct_before_onset", "amb_pct_after_end"])
        w.writerows(piece_rows)

    return {
        "n_pieces": len(piece_rows),
        "episode_amb_pct_before_median": float(np.nanmedian(pct_befores)),
        "episode_amb_pct_after_median": float(np.nanmedian(pct_afters)),
        "episode_n_above_50": n_above, "episode_n_below_50": n_below,
        "episode_sign_test_p": float(ep_p), "episode_wilcoxon_p": ep_wilcoxon_p,
        "median_max_drift_gain_sec": float(np.nanmedian(ep_gains)),
        "n_peak_lag_positive": n_pos, "n_peak_lag_negative": n_neg, "n_peak_lag_zero": n_zero,
        "sign_test_p": float(sign_p),
        "median_peak_lag_sec": float(np.nanmedian(peak_lags_sec)),
        "median_peak_lag_sec_unrectified": float(np.nanmedian(raw_lags_sec)),
        "median_peak_corr": float(np.nanmedian(peak_corrs)),
        "median_null_percentile": float(np.nanmedian(null_pcts)),
        "events_before": tot_before, "events_after": tot_after,
        "events_total_drift": tot_trigger, "before_after_binom_p": float(ba_p),
    }


def run_p1_week4(cfg, out_dir, log):
    hop = cfg["feature"]["frame_hop_sec"]
    song_package = _resolve(cfg["data"]["song_package"])
    performance_wav = _resolve(cfg["data"]["performance_wav"])
    max_sec = cfg["data"].get("score_max_sec", 90.0)
    recording_len = cfg["data"]["recording_length_sec"]

    score_chroma = build_score_chroma_v2(song_package, hop_sec=hop, max_sec=max_sec,
                                          recording_len=recording_len)
    perf_chroma, perf_dur = audio_chroma(performance_wav, hop=hop)
    log(f"연주 길이={perf_dur:.2f}s ({len(perf_chroma)} 프레임), "
        f"스코어 프레임수={len(score_chroma)}")

    window = int(round(cfg["feature"]["ambiguity_window_sec"] / hop))
    exclude = int(round(cfg["feature"]["ambiguity_exclude_sec"] / hop))
    merge_frames = int(round(cfg.get("v7", {}).get("merge_sec", 2.0) / hop))
    sim, n_windows = build_similarity_matrix(
        score_chroma, window, transpose_invariant=cfg["feature"]["transpose_invariant"]
    )
    maxsim = max_sim_ambiguity(sim, n_windows, exclude)
    v7cfg = cfg.get("v7", {})
    ent_amb, _ = cluster_entropy_ambiguity(
        sim, n_windows, exclude, merge_frames,
        temperature=v7cfg.get("temperature", 0.001),
        peak_rel_threshold=v7cfg.get("peak_rel_threshold", 0.01),
    )

    band_width = int(round(cfg["tracker"]["band_width_sec"] / hop))
    dtw = OnlineDTW(perf_chroma, score_chroma, band_width=band_width,
                     max_run_count=cfg["tracker"].get("max_run_count", 3))
    dtw.run()
    real_path = dtw.real_path()
    est_j_of_t = {}
    for t, j in real_path:
        est_j_of_t[t] = j

    # GT 가정: true_score_time == perf_time (onset-detection으로 별도 검증됨,
    # ENGINEERING_LOG.md "objective 검증" 절 — 평균오차 85ms, 드리프트 없음)
    rows = []
    for t, j in est_j_of_t.items():
        perf_time = t * hop
        if recording_len is not None and perf_time > recording_len:
            continue
        est_score_time = j * hop
        true_score_time = perf_time
        error = abs(est_score_time - true_score_time)
        true_j = int(round(true_score_time / hop))
        if not (exclude <= true_j < n_windows - exclude):
            continue
        if np.isnan(ent_amb[true_j]):
            continue
        rows.append((perf_time, error, maxsim[true_j], ent_amb[true_j]))

    with open(os.path.join(out_dir, "frame_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perf_time_sec", "error_sec", "maxsim", "v7_effective_candidates"])
        w.writerows(rows)

    err = np.array([r[1] for r in rows])
    ms = np.array([r[2] for r in rows])
    ea = np.array([r[3] for r in rows])
    r_max, p_max = spearmanr(ms, err)
    r_v7, p_v7 = spearmanr(ea, err)
    log(f"n={len(rows)}  오차중앙값={np.median(err):.3f}s")
    log(f"rho(maxsim,오차)={r_max:+.3f} p={p_max:.3g}   rho(v7,오차)={r_v7:+.3f} p={p_v7:.3g}")

    primary = cfg["ambiguity_metric"]
    rho_primary, p_primary = (r_max, p_max) if primary == "maxsim" else (r_v7, p_v7)

    return {
        "rho": float(rho_primary), "p_value": float(p_primary),
        "rho_maxsim": float(r_max), "p_maxsim": float(p_max),
        "rho_v7": float(r_v7), "p_v7": float(p_v7),
        "median_error_sec": float(np.median(err)),
        "n_frames": len(rows),
        "gt_assumption": "true_score_time == perf_time (onset-detection verified, ~85ms mean error)",
    }


def run_p2_week4_gap(cfg, out_dir, log):
    """P2 week4 외적 타당성: bansanka/DLBA 실제 기타 녹음에서도 온라인 vs 오프라인
    GAP이 모호성과 양의 상관을 보이는지 확인. ASAP(run_p2_asap_gap)이 성립을 먼저
    확인한 뒤 진행 — FEEDBACK.md "P2 설계" 절 "ASAP(대규모) 먼저 → 성립하면 week4".

    ⚠️ `open_end=True`를 쓴다 — week4 녹음은 스코어 전체(원곡)의 접두어(prefix)일 뿐이라
    (bansanka: 61.58초 녹음 vs 90초로 자른 스코어), 끝점을 스코어 끝으로 강제하면 안 됨
    (offline_dtw.py 설계 근거 참조). 시작은 (0,0)로 고정 — onset-detection으로 이미
    검증된 GT 가정(true_score_time == perf_time)과 일치."""
    hop = cfg["feature"]["frame_hop_sec"]
    song_package = _resolve(cfg["data"]["song_package"])
    performance_wav = _resolve(cfg["data"]["performance_wav"])
    max_sec = cfg["data"].get("score_max_sec", 90.0)
    recording_len = cfg["data"]["recording_length_sec"]

    score_chroma = build_score_chroma_v2(song_package, hop_sec=hop, max_sec=max_sec,
                                          recording_len=recording_len)
    perf_chroma, perf_dur = audio_chroma(performance_wav, hop=hop)
    log(f"연주 길이={perf_dur:.2f}s ({len(perf_chroma)} 프레임), "
        f"스코어 프레임수={len(score_chroma)}")

    window = int(round(cfg["feature"]["ambiguity_window_sec"] / hop))
    exclude = int(round(cfg["feature"]["ambiguity_exclude_sec"] / hop))
    sim, n_windows = build_similarity_matrix(
        score_chroma, window, transpose_invariant=cfg["feature"]["transpose_invariant"]
    )
    maxsim = max_sim_ambiguity(sim, n_windows, exclude)

    band_width = int(round(cfg["tracker"]["band_width_sec"] / hop))
    online = OnlineDTW(perf_chroma, score_chroma, band_width=band_width,
                        max_run_count=cfg["tracker"].get("max_run_count", 3))
    online.run()
    online_j_of_t = {}
    for t, j in online.real_path():
        online_j_of_t[t] = j

    # ⚠️ offline에도 온라인과 '같은' band_width를 준다 — offline_dtw.py "band_width 옵션"
    # 절 참조: week4는 국소비용이 전반적으로 약해서(진짜 대응도 코사인유사도 ~0.5) 밴드
    # 없이 열어두면 짧은 반복 구간에 눌러앉는 새 퇴화가 생김(bansanka에서 실제 확인,
    # ENGINEERING_LOG.md 참조). 같은 band를 쓰면 "정보량"이 아니라 "인과성/탐욕적
    # 의사결정 여부"만 비교 대상으로 남는다 — P2가 원래 분리하려는 바로 그 변수.
    t0 = time.time()
    offline = OfflineDTW(perf_chroma, score_chroma, open_end=True, band_width=band_width)
    offline.run()
    offline_runtime = time.time() - t0
    offline_j_of_t = offline.j_of_t()
    log(f"offline 실행시간={offline_runtime:.1f}s  offline 끝점 스코어프레임={offline.end_j_}"
        f" (참고: 연주 마지막 프레임이 스코어 {offline.end_j_*hop:.2f}s에 대응)")

    # GT 가정: true_score_time == perf_time (onset-detection으로 별도 검증됨)
    rows = []
    for t in range(len(perf_chroma)):
        if t not in online_j_of_t or t not in offline_j_of_t:
            continue
        perf_time = t * hop
        if recording_len is not None and perf_time > recording_len:
            continue
        true_score_time = perf_time
        online_err = abs(online_j_of_t[t] * hop - true_score_time)
        offline_err = abs(offline_j_of_t[t] * hop - true_score_time)
        true_j = int(round(true_score_time / hop))
        if not (exclude <= true_j < n_windows - exclude):
            continue
        if np.isnan(maxsim[true_j]):
            continue
        gap = online_err - offline_err
        rows.append((perf_time, online_err, offline_err, gap, maxsim[true_j]))

    with open(os.path.join(out_dir, "frame_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perf_time_sec", "online_error_sec", "offline_error_sec", "gap_sec", "maxsim"])
        w.writerows(rows)

    online_err = np.array([r[1] for r in rows])
    offline_err = np.array([r[2] for r in rows])
    gap = np.array([r[3] for r in rows])
    ms = np.array([r[4] for r in rows])

    r_gap, p_gap = spearmanr(ms, gap)
    r_online, p_online = spearmanr(ms, online_err)
    r_offline, p_offline = spearmanr(ms, offline_err)
    log(f"n={len(rows)}")
    log(f"online 오차중앙값={np.median(online_err):.3f}s  offline 오차중앙값={np.median(offline_err):.3f}s")
    log(f"rho(maxsim,GAP)={r_gap:+.3f} p={p_gap:.3g}  "
        f"rho(maxsim,online_err)={r_online:+.3f} p={p_online:.3g}  "
        f"rho(maxsim,offline_err)={r_offline:+.3f} p={p_offline:.3g}")

    return {
        "rho": float(r_gap), "p_value": float(p_gap),
        "rho_gap": float(r_gap), "p_gap": float(p_gap),
        "rho_online_err": float(r_online), "p_online_err": float(p_online),
        "rho_offline_err": float(r_offline), "p_offline_err": float(p_offline),
        "online_median_error_sec": float(np.median(online_err)),
        "offline_median_error_sec": float(np.median(offline_err)),
        "offline_runtime_sec": float(offline_runtime),
        "offline_end_j": int(offline.end_j_),
        "n_frames": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 42))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "results", f"{cfg['experiment']}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log(f"=== {cfg['experiment']} ({cfg['kind']}) ===")
    log(f"config: {config_path}")

    if cfg["kind"] == "p1_asap":
        metrics = run_p1_asap(cfg, out_dir, log)
    elif cfg["kind"] == "p1_week4":
        metrics = run_p1_week4(cfg, out_dir, log)
    elif cfg["kind"] == "p2_asap_gap":
        metrics = run_p2_asap_gap(cfg, out_dir, log)
    elif cfg["kind"] == "p2_week4_gap":
        metrics = run_p2_week4_gap(cfg, out_dir, log)
    elif cfg["kind"] == "p3_asap_order":
        metrics = run_p3_asap_order(cfg, out_dir, log)
    else:
        raise ValueError(f"unknown kind: {cfg['kind']}")

    metrics.update({
        "experiment": cfg["experiment"],
        "kind": cfg["kind"],
        "ambiguity_metric": cfg["ambiguity_metric"],
        "git_commit": _git_commit(),
        "seed": cfg.get("seed", 42),
        "timestamp": timestamp,
    })

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    with open(os.path.join(out_dir, "environment.txt"), "w", encoding="utf-8") as f:
        import numpy, scipy, librosa
        f.write(f"python={platform.python_version()}\n")
        f.write(f"platform={platform.platform()}\n")
        f.write(f"numpy={numpy.__version__}\n")
        f.write(f"scipy={scipy.__version__}\n")
        f.write(f"librosa={librosa.__version__}\n")
        f.write(f"git_commit={_git_commit()}\n")
        f.write(f"run_at={datetime.now().isoformat()}\n")

    log(f"\n결과 저장: {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
