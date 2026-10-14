#!/usr/bin/env python3
"""Frozen CP-WTC robustness audit of the continuous P3-TDR endpoint."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import p3_td_transfer as transfer
from score_following.features import load_notes, notes_to_chroma_stream

MANIFEST = ROOT / "results/p3_cp_wtc_preflight/manifest.json"
OUT = ROOT / "results/p3_tdr_cp_wtc_validation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_alignment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = np.loadtxt(path)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2:
        raise ValueError(f"invalid alignment: {path}")
    score_clock, perf_clock = points[:, 0] * .5, points[:, 1]
    if np.any(np.diff(score_clock) <= 0) or np.any(np.diff(perf_clock) <= 0):
        raise ValueError(f"non-monotone alignment: {path}")
    return score_clock, perf_clock


def load_case(case: dict, hop: float):
    import partitura as pt
    midi = ROOT / case["midi"]
    score_path = ROOT / case["score"]
    alignment_path = ROOT / case["alignment"]
    if any((digest(path) != case[key]) for path, key in (
        (midi, "midi_sha256"), (score_path, "score_sha256"),
        (alignment_path, "alignment_sha256"))):
        raise AssertionError("CP-WTC file changed after preflight")
    score_clock, perf_clock = load_alignment(alignment_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pt.load_musicxml(str(score_path))
        score_array = pt.score.unfold_part_maximal(parsed[0], ignore_leaps=False).note_array()
    score_notes = [(float(row["onset_beat"]) * .5,
                    float(row["onset_beat"] + row["duration_beat"]) * .5,
                    int(row["pitch"])) for row in score_array]
    score_duration = max(note[1] for note in score_notes)
    performance_notes = load_notes(midi)
    performance_duration = max(note[1] for note in performance_notes)
    # The alignment spans the complete recording, while this MIDI contains the
    # annotated first eight bars. Calibrate only on the represented interval.
    covered_start = max(0.0, float(perf_clock[0]))
    covered_stop = min(performance_duration, float(perf_clock[-1]))
    score_start, score_stop = np.interp([covered_start, covered_stop], perf_clock, score_clock)
    score_hop = hop * (score_stop - score_start) / (covered_stop - covered_start)
    u = notes_to_chroma_stream(performance_notes, performance_duration, hop)
    v = notes_to_chroma_stream(score_notes, score_duration, score_hop)
    frame_times = np.arange(len(u)) * hop
    truth_score_time = np.interp(frame_times, perf_clock, score_clock,
                                 left=score_clock[0], right=score_clock[-1])
    truth = truth_score_time / score_hop
    return u, v, score_hop, truth, min(performance_duration, float(perf_clock[-1]))


def main() -> None:
    batik = json.loads((ROOT / "results/p3_tdr_batik_validation/result.json").read_text(encoding="utf8"))
    if batik["p3_tdr_confirmed"]:
        raise SystemExit("CP-WTC arbitration is only defined after the preserved Batik failure")
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = []
    for row in manifest["rows"]:
        if row["eligible"]:
            cases.append(dict(row, unit=row["piece"]))
    result = transfer.run_cases(
        cases, load_case, "unit",
        lambda case, available: np.asarray([15.0]),
        OUT, "CP-WTC external robustness after preserved Batik failure",
        lambda units, landmarks: units >= 12 and landmarks >= 35,
    )
    required = ("coverage", "replay_and_manipulation", "risk_curve_strictly_increasing",
                "positive_risk_rho_share_ge_0_80", "ten_sec_risk_ge_0_25",
                "ten_sec_risk_ci_lower_gt_0")
    result["p3_tdr_risk_gates"] = {key: result["gates"][key] for key in required}
    result["p3_tdr_cp_wtc_supported"] = bool(all(result["p3_tdr_risk_gates"].values()))
    result["batik_failure_preserved"] = True
    result["dataset_limitations"] = manifest["known_limitations"]
    result["original_natural_p3_confirmed"] = False
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"counts": result["counts"], "mean_curve": result["mean_curve"],
                      "risk_ci": result["ten_sec_risk_bootstrap_95"],
                      "risk_gates": result["p3_tdr_risk_gates"],
                      "supported": result["p3_tdr_cp_wtc_supported"]}, indent=2))


if __name__ == "__main__":
    main()
