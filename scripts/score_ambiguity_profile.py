"""Score-level ambiguity statistics computed from the score alone.

A_work must not be a mean over evaluated rows: which frames are evaluated is set by
the annotated beat times, so that mean would read ground-truth timing. These
statistics come from the score's own maxsim profile over every defined state, are
available before the performance starts, and involve no annotation at all.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.confidence_audit import blocked_maxsim
from score_following.features import load_notes, notes_to_chroma_stream

CONFIG = ROOT / "configs/confidence_audit_v2.json"
OUT = ROOT / "results/score_ambiguity_profile.json"


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    data = ROOT / cfg["asap_root"]
    source = ROOT / cfg["source_manifest"]
    with source.open(encoding="utf-8-sig", newline="") as f:
        cases = [r for r in csv.DictReader(f) if r["role"] == cfg["role"]]
    hs = cfg["score_hop_sec"]
    profile = {}
    with threadpool_limits(limits=cfg["blas_threads"]):
        for case in cases:
            key = case["score_sha256"]
            if key in profile:
                continue
            notes = load_notes(data / case["score"])
            chroma = notes_to_chroma_stream(notes, max(n[1] for n in notes), hs)
            amb, silent = blocked_maxsim(chroma, int(round(cfg["ambiguity_window_score_sec"]/hs)),
                                         int(round(cfg["ambiguity_exclude_score_sec"]/hs)),
                                         cfg["matrix_block_rows"])
            finite = amb[np.isfinite(amb)]
            profile[key] = {"score": case["score"], "states": int(len(amb)),
                            "defined_states": int(len(finite)),
                            "silent_contexts": int(silent),
                            "mean": float(finite.mean()), "median": float(np.median(finite)),
                            "p25": float(np.quantile(finite, .25)),
                            "p75": float(np.quantile(finite, .75))}
            print(f"{case['score']}: states={len(amb)} mean={profile[key]['mean']:.4f}", flush=True)
    OUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(profile)} distinct scores -> {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
