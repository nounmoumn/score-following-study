#!/usr/bin/env python3
"""Reproducible adapter for Nakamura's official symbolic alignment tool.

Outputs are automatic anchor candidates only. They are never natural-P3 GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external_tools/AlignmentTool_v240109.zip"
TOOL = ROOT / "data/external_tools/AlignmentTool_v240109/AlignmentTool"
PROGRAMS = TOOL / "Programs"
ARCHIVE_SHA256 = "cf75af54435c6ad83a7b578b691724866f6df5701e8529d86ae2a3db1e85bddd"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def windows_to_wsl(path):
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{remainder}"


def verify_tool():
    if not ARCHIVE.exists() or sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("Nakamura source archive is absent or its frozen SHA-256 differs")
    required = ("midi2pianoroll", "SprToFmt3x", "Fmt3xToHmm", "ScorePerfmMatcher", "ErrorDetection", "RealignmentMOHMM", "MatchToCorresp")
    missing = [name for name in required if not (PROGRAMS / name).exists()]
    if missing:
        raise RuntimeError(f"Nakamura binaries missing: {missing}")
    return required


def run_alignment(score_midi, performance_midi, output_directory):
    required = verify_tool()
    output = Path(output_directory).resolve()
    results_root = (ROOT / "results").resolve()
    if results_root not in output.parents:
        raise ValueError("Alignment output must be a new directory under workspace/results")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(score_midi, output / "ref.mid")
    shutil.copy2(performance_midi, output / "perf.mid")
    p = windows_to_wsl(PROGRAMS)
    o = windows_to_wsl(output)
    commands = [
        f"{shlex.quote(p + '/midi2pianoroll')} 0 ref",
        f"{shlex.quote(p + '/midi2pianoroll')} 0 perf",
        f"{shlex.quote(p + '/SprToFmt3x')} ref_spr.txt ref_fmt3x.txt",
        f"{shlex.quote(p + '/Fmt3xToHmm')} ref_fmt3x.txt ref_hmm.txt",
        f"{shlex.quote(p + '/ScorePerfmMatcher')} ref_hmm.txt perf_spr.txt perf_pre_match.txt 0.001",
        f"{shlex.quote(p + '/ErrorDetection')} ref_fmt3x.txt ref_hmm.txt perf_pre_match.txt perf_err_match.txt 0",
        f"{shlex.quote(p + '/RealignmentMOHMM')} ref_fmt3x.txt ref_hmm.txt perf_err_match.txt perf_realigned_match.txt 0.3",
        "cp perf_realigned_match.txt perf_match.txt",
        f"{shlex.quote(p + '/MatchToCorresp')} perf_match.txt ref_spr.txt perf_corresp.txt",
    ]
    subprocess.run(["wsl.exe", "bash", "-lc", f"set -euo pipefail; cd {shlex.quote(o)}; " + "; ".join(commands)], check=True)
    corresp = output / "perf_corresp.txt"
    if not corresp.exists():
        raise RuntimeError("Alignment completed without a correspondence file")
    provenance = {
        "archive_sha256": ARCHIVE_SHA256,
        "score_sha256": sha256(score_midi),
        "performance_sha256": sha256(performance_midi),
        "program_sha256": {name: sha256(PROGRAMS / name) for name in required},
        "corresp_sha256": sha256(corresp),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")
    return corresp


def parse_corresp(path):
    matched = []
    missing_score = []
    extra_performance = []
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            if line.startswith("//") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"Malformed corresp row: {line!r}")
            align_missing, reference_missing = fields[0] == "*", fields[5] == "*"
            if align_missing:
                missing_score.append({"reference_id": fields[5], "reference_time": float(fields[6]), "reference_pitch": int(fields[8])})
            elif reference_missing:
                extra_performance.append({"performance_id": fields[0], "performance_time": float(fields[1]), "performance_pitch": int(fields[3])})
            else:
                matched.append({
                    "performance_id": fields[0],
                    "performance_time": float(fields[1]),
                    "performance_pitch": int(fields[3]),
                    "reference_id": fields[5],
                    "reference_time": float(fields[6]),
                    "reference_pitch": int(fields[8]),
                })
    return matched, missing_score, extra_performance


def clock_anchors(matched):
    by_time = {}
    for row in matched:
        by_time.setdefault(row["performance_time"], []).append(row["reference_time"])
    return np.asarray([(time, float(np.median(by_time[time]))) for time in sorted(by_time)], dtype=float)


def audit_corresp(path):
    matched, missing, extra = parse_corresp(path)
    anchors = clock_anchors(matched)
    return {
        "matched_notes": len(matched),
        "missing_score_notes": len(missing),
        "extra_performance_notes": len(extra),
        "unique_clock_anchors": len(anchors),
        "clock_monotone": bool(len(anchors) and np.all(np.diff(anchors[:, 1]) >= 0)),
        "first_anchor": anchors[0].tolist() if len(anchors) else None,
        "last_anchor": anchors[-1].tolist() if len(anchors) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_midi", type=Path)
    parser.add_argument("performance_midi", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    corresp = run_alignment(args.score_midi, args.performance_midi, args.output_directory)
    result = audit_corresp(corresp)
    (args.output_directory / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
