# -*- coding: utf-8 -*-
"""
관계식 도출 — 저장된 P2 실행 산출물(frame_results.csv)에서 e(A)와 K(A)를 적합한다.

    python scripts/derive_relations.py                     # 최신 P2 결과 자동 탐색
    python scripts/derive_relations.py --asap <dir> --week4 <dir> <dir>

왜 P2 산출물을 쓰는가: P2의 frame_results.csv에만 프레임별 (모호성, 온라인오차,
오프라인오차, GAP)이 **한 줄에 같이** 들어있어서, 같은 프레임 위에서 온라인/오프라인
관계식을 나란히 적합할 수 있다(P1 산출물엔 오프라인 오차가 없다).

산출: results/paper/table_relations_static.csv, table_relations_gain.csv + 콘솔 요약.
근거·해석은 ENGINEERING_LOG.md "관계식 도출" 절에 기록할 것.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from score_following.relations import (  # noqa: E402
    binned_medians, fit_gain_from_traces, fit_gain_per_piece, fit_static_forms,
)

HOP = 0.1


def _latest(pattern):
    dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results", pattern)))
    return dirs[-1] if dirs else None


def load_asap(run_dir):
    """P2 ASAP frame_results.csv -> {label: (amb, online_err, offline_err)}"""
    path = os.path.join(run_dir, "frame_results.csv")
    per_piece = defaultdict(lambda: ([], [], []))
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a, on, off = per_piece[row["label"]]
            a.append(float(row["ambiguity"]))
            on.append(float(row["online_error_sec"]))
            off.append(float(row["offline_error_sec"]))
    return {k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in per_piece.items()}


def load_week4(run_dir):
    """P2 week4 frame_results.csv -> (amb, online_err, offline_err)"""
    path = os.path.join(run_dir, "frame_results.csv")
    a, on, off = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a.append(float(row["maxsim"]))
            on.append(float(row["online_error_sec"]))
            off.append(float(row["offline_error_sec"]))
    return np.array(a), np.array(on), np.array(off)


def report_static(name, amb, err, out_rows, n_bins=10):
    """구간 중앙값으로 요약한 뒤 세 함수 형태를 비교."""
    bins = binned_medians(amb, err, n_bins=n_bins)
    if len(bins) < 4:
        print(f"  {name}: 구간이 부족해 건너뜀")
        return
    bA = np.array([b[0] for b in bins])
    bE = np.array([b[1] for b in bins])
    forms = fit_static_forms(bA, bE)
    best = max(forms, key=lambda d: (d["r2"] if np.isfinite(d["r2"]) else -np.inf))
    print(f"  {name}  (구간 {len(bins)}개, 모호성 {bA.min():.3f}~{bA.max():.3f}, "
          f"오차 {bE.min():.3f}~{bE.max():.3f}s)")
    for f in forms:
        mark = " <-- 최적" if f is best else ""
        print(f"    {f['form']:24s} a={f['a']:+.4f}  b={f['b']:+.4f}  r2={f['r2']:.4f}{mark}")
        out_rows.append([name, f["form"], f["a"], f["b"], f["r2"], len(bins)])
    return best


def report_gain(name, traces, out_rows, n_bins=5):
    """모호성 구간별 1차 제어 모델 de/dt = -K*e + c 적합."""
    fits = fit_gain_from_traces(traces, HOP, n_bins=n_bins)
    if not fits:
        print(f"  {name}: 게인 적합 불가(표본 부족)")
        return
    print(f"  {name}")
    print(f"    {'모호성':>8s} {'K(1/s)':>9s} {'tau=1/K(s)':>11s} {'c(s/s)':>9s} "
          f"{'r2':>6s} {'n':>7s}")
    for f in fits:
        tau = f["tau_sec"]
        tau_str = "inf" if not np.isfinite(tau) else f"{tau:.2f}"
        print(f"    {f['amb_median']:8.3f} {f['K']:9.4f} {tau_str:>11s} {f['c']:9.4f} "
              f"{f['r2']:6.3f} {f['n']:7d}")
        out_rows.append([name, f["amb_median"], f["K"], tau, f["c"], f["r2"], f["n"]])
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asap", default=None)
    ap.add_argument("--week4", nargs="*", default=None)
    args = ap.parse_args()

    asap_dir = args.asap or _latest("p2_asap_maxsim_*")
    week4_dirs = args.week4 or [d for d in [_latest("p2_week4_bansanka_*"),
                                            _latest("p2_week4_dlba_*")] if d]

    static_rows, gain_rows = [], []

    print("=" * 78)
    print("[1] 정적 관계 e(A) — 어떤 함수 형태가 맞는가 (구간 중앙값 기준)")
    print("=" * 78)

    if asap_dir:
        print(f"\nASAP ({os.path.basename(asap_dir)})")
        pieces = load_asap(asap_dir)
        amb_all = np.concatenate([v[0] for v in pieces.values()])
        on_all = np.concatenate([v[1] for v in pieces.values()])
        off_all = np.concatenate([v[2] for v in pieces.values()])
        report_static("ASAP online", amb_all, on_all, static_rows)
        report_static("ASAP offline", amb_all, off_all, static_rows)
        report_static("ASAP GAP", amb_all, on_all - off_all, static_rows)

    for d in week4_dirs:
        tag = "bansanka" if "bansanka" in d else "dlba"
        print(f"\nweek4/{tag} ({os.path.basename(d)})")
        amb, on, off = load_week4(d)
        report_static(f"week4-{tag} online", amb, on, static_rows)
        report_static(f"week4-{tag} offline", amb, off, static_rows)

    print()
    print("=" * 78)
    print("[2] 동적 관계 — 1차 제어 모델  de/dt = -K(A)*e + c")
    print("    K = 관측이 오차를 되돌리는 보정 게인(1/초). tau=1/K는 회복 시간상수.")
    print("    가설(§2 메커니즘): 모호성 A가 커지면 K -> 0 (개루프 적분기가 됨)")
    print("=" * 78)

    if asap_dir:
        traces = [(v[0], v[1]) for v in pieces.values()]
        print()
        report_gain("ASAP online (곡별 궤적 합산)", traces, gain_rows)
        traces_off = [(v[0], v[2]) for v in pieces.values()]
        report_gain("ASAP offline (대조)", traces_off, gain_rows)

    for d in week4_dirs:
        tag = "bansanka" if "bansanka" in d else "dlba"
        amb, on, off = load_week4(d)
        print()
        report_gain(f"week4-{tag} online", [(amb, on)], gain_rows, n_bins=4)

    print()
    print("=" * 78)
    print("[3] 試行 2 — 곡 내 짝바교(모호성 닮은 절반 vs 높은 절반)")
    print("    1자 적합이 여러 곡 합산에서 실패해서, 미분 전 이동평걱 + 곡 내 짝바교로 재시도")
    print("=" * 78)
    if asap_dir:
        per = fit_gain_per_piece([(v[0], v[1]) for v in pieces.values()], HOP,
                                  smooth_frames=20)
        n_hi_lower = sum(1 for d in per
                          if np.isfinite(d.get("K_hi", np.nan))
                          and np.isfinite(d.get("K_lo", np.nan))
                          and d["K_hi"] < d["K_lo"])
        n_valid = sum(1 for d in per
                       if np.isfinite(d.get("K_hi", np.nan))
                       and np.isfinite(d.get("K_lo", np.nan)))
        K_lo = np.array([d.get("K_lo", np.nan) for d in per], dtype=float)
        K_hi = np.array([d.get("K_hi", np.nan) for d in per], dtype=float)
        r2s = np.array([d.get("r2_lo", np.nan) for d in per]
                        + [d.get("r2_hi", np.nan) for d in per], dtype=float)
        print(f"  ASAP 곡 {n_valid}개 짝바교")
        print(f"    K(모호성 낮은 절반) 중앙값 = {np.nanmedian(K_lo):+.4f} /s")
        print(f"    K(모호성 높은 절반) 중앙값 = {np.nanmedian(K_hi):+.4f} /s")
        print(f"    K_hi < K_lo 인 곡: {n_hi_lower}/{n_valid}")
        print(f"    적합 r2 중앙값 = {np.nanmedian(r2s):.3f} "
              f"(낮으면 1자 모델이 여전히 안 맞는 것)")
        if n_valid > 0:
            from scipy.stats import binomtest
            print(f"    부호검정 p = {binomtest(n_hi_lower, n_valid, 0.5).pvalue:.3g}")
        for d in per:
            gain_rows.append([f"ASAP piece{d['piece_idx']} paired",
                               d["amb_lo"], d.get("K_lo"), d.get("tau_lo"),
                               d.get("c_lo"), d.get("r2_lo"), d["n"]])
            gain_rows.append([f"ASAP piece{d['piece_idx']} paired_hi",
                               d["amb_hi"], d.get("K_hi"), d.get("tau_hi"),
                               d.get("c_hi"), d.get("r2_hi"), d["n"]])

    out_dir = os.path.join(REPO_ROOT, "results", "paper")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table_relations_static.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "form", "a", "b", "r2", "n_bins"])
        w.writerows(static_rows)
    with open(os.path.join(out_dir, "table_relations_gain.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "ambiguity_median", "K_per_sec", "tau_sec", "c_sec_per_sec",
                     "r2", "n_frames"])
        w.writerows(gain_rows)
    print(f"\n표 저장: {out_dir}/table_relations_static.csv, table_relations_gain.csv")


if __name__ == "__main__":
    main()
