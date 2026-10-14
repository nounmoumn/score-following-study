# -*- coding: utf-8 -*-
"""
한 번에 논문 표를 재현한다.

    python scripts/reproduce_paper.py

configs/paper/의 모든 설정을 순서대로 돌리고, results/paper/에 표 CSV를 모은다.
개별 실험만 다시 돌리고 싶으면:

    python -m score_following.run --config configs/paper/p1_asap_maxsim.yaml
"""
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from score_following.run import main as run_experiment  # noqa: E402

CONFIGS = [
    "configs/paper/p1_asap_maxsim.yaml",
    "configs/paper/p1_asap_v7.yaml",
    "configs/paper/p1_week4_bansanka.yaml",
    "configs/paper/p1_week4_dlba.yaml",
    # 2026-09-08/17 추가 (P2·P3). ⚠️ P2 ASAP과 P3는 오프라인 전역 DTW·순환이동
    # 귀무분포 때문에 각각 10분~1시간 단위로 걸린다 — 빠르게 P1만 돌리려면
    # --only p1_asap_maxsim p1_week4_bansanka 처럼 지정할 것.
    "configs/paper/p2_asap_maxsim.yaml",
    "configs/paper/p2_week4_bansanka.yaml",
    "configs/paper/p2_week4_dlba.yaml",
    "configs/paper/p3_asap_order.yaml",
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                         help="특정 config만 돌리려면 파일명(확장자 없이) 지정, 예: --only p1_asap_maxsim")
    args = parser.parse_args()

    results = {}
    for cfg_rel in CONFIGS:
        name = os.path.splitext(os.path.basename(cfg_rel))[0]
        if args.only and name not in args.only:
            continue
        cfg_path = os.path.join(REPO_ROOT, cfg_rel)
        print(f"\n{'='*70}\n실행: {cfg_rel}\n{'='*70}")
        sys.argv = ["run.py", "--config", cfg_path]
        results[name] = run_experiment()

    out_dir = os.path.join(REPO_ROOT, "results", "paper")
    os.makedirs(out_dir, exist_ok=True)

    asap_rows = [(k, v) for k, v in results.items() if k.startswith("p1_asap")]
    if asap_rows:
        with open(os.path.join(out_dir, "table_p1_asap.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "ambiguity_metric", "rho", "z", "p_value",
                        "n_healthy_pieces", "shuffle_control_z"])
            for name, m in asap_rows:
                w.writerow([name, m["ambiguity_metric"], m["rho"], m.get("z"), m["p_value"],
                            m.get("n_healthy_pieces"), m.get("shuffle_control_z")])

    week4_rows = [(k, v) for k, v in results.items() if k.startswith("p1_week4")]
    if week4_rows:
        with open(os.path.join(out_dir, "table_p1_week4.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "ambiguity_metric", "rho_maxsim", "p_maxsim",
                        "rho_v7", "p_v7", "median_error_sec", "n_frames"])
            for name, m in week4_rows:
                w.writerow([name, m["ambiguity_metric"], m["rho_maxsim"], m["p_maxsim"],
                            m["rho_v7"], m["p_v7"], m["median_error_sec"], m["n_frames"]])

    p2_rows = [(k, v) for k, v in results.items() if k.startswith("p2_")]
    if p2_rows:
        with open(os.path.join(out_dir, "table_p2.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "rho_gap", "p_value", "rho_online_err", "rho_offline_err",
                        "n_healthy_pieces", "shuffle_control_z"])
            for name, m in p2_rows:
                w.writerow([name, m.get("rho"), m.get("p_value"),
                            m.get("rho_online_err"), m.get("rho_offline_err"),
                            m.get("n_healthy_pieces"), m.get("shuffle_control_z")])

    p3_rows = [(k, v) for k, v in results.items() if k.startswith("p3_")]
    if p3_rows:
        with open(os.path.join(out_dir, "table_p3.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "n_pieces", "episode_amb_pct_before_median",
                        "episode_amb_pct_after_median", "episode_sign_test_p",
                        "episode_wilcoxon_p", "median_peak_lag_sec", "sign_test_p",
                        "events_before", "events_after", "before_after_binom_p"])
            for name, m in p3_rows:
                w.writerow([name, m.get("n_pieces"),
                            m.get("episode_amb_pct_before_median"),
                            m.get("episode_amb_pct_after_median"),
                            m.get("episode_sign_test_p"), m.get("episode_wilcoxon_p"),
                            m.get("median_peak_lag_sec"), m.get("sign_test_p"),
                            m.get("events_before"), m.get("events_after"),
                            m.get("before_after_binom_p")])

    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items()}, f, ensure_ascii=False, indent=2)

    # 관계식 표는 P2 산출물을 입력으로 쓰므로 실험 실행 뒤에 한 번 더 돌린다
    if p2_rows or args.only is None:
        print("\n관계식 도출(derive_relations.py) 실행 —")
        os.system(f'"{sys.executable}" "{os.path.join(REPO_ROOT, "scripts", "derive_relations.py")}"')

    print(f"\n논문 표: {out_dir}")


if __name__ == "__main__":
    main()
