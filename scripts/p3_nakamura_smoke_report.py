#!/usr/bin/env python3
"""Report source-compatible ASAP smoke metrics without claiming independent GT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scripts import p3_nakamura_adapter as adapter
except ImportError:
    import p3_nakamura_adapter as adapter

ROOT = adapter.ROOT
RUN = ROOT / "results/p3_nakamura_adapter_smoke"


def main():
    matched, _, _ = adapter.parse_corresp(RUN / "perf_corresp.txt")
    anchors = adapter.clock_anchors(matched)
    performance = np.loadtxt(ROOT / "asap_dataset/Bach/Fugue/bwv_866/SOLOM02_annotations.txt", usecols=0)
    score = np.loadtxt(ROOT / "asap_dataset/Bach/Fugue/bwv_866/midi_score_annotations.txt", usecols=0)
    support = (performance >= anchors[0, 0]) & (performance <= anchors[-1, 0])
    residual = np.abs(np.interp(performance[support], anchors[:, 0], anchors[:, 1]) - score[support])
    result = {
        "scope": "ASAP source-compatible engineering smoke test",
        "work": "Bach/Fugue/bwv_866/SOLOM02.mid",
        "evaluated_beats": int(np.sum(support)),
        "outside_support": int(np.sum(~support)),
        "median_abs_score_sec": float(np.median(residual)),
        "p95_abs_score_sec": float(np.percentile(residual, 95)),
        "max_abs_score_sec": float(np.max(residual)),
        "audit": adapter.audit_corresp(RUN / "perf_corresp.txt"),
        "independent_validation": False,
        "limitation": "ASAP annotations were produced from an automatic alignment workflow with manual correction; replaying Nakamura-family alignment is compatibility evidence only.",
        "natural_p3_ground_truth": False,
    }
    (RUN / "beat_compatibility.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
