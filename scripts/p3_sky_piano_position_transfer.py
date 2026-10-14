#!/usr/bin/env python3
"""Frozen position/sign feasibility transfer to SKY-Piano public samples."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.temporal import healthy_before

ACQUISITION = ROOT / "results/p3_sky_piano_sample_preflight/manifest.json"
ADAPTER = ROOT / "results/p3_sky_piano_clock_adapter/manifest.json"
OUT = ROOT / "results/p3_sky_piano_position_transfer"
WINDOW_SEC = 12.0
HORIZON_SEC = 0.5
RADIUS = 0.5211881394162992
CAP = 5.0
LANDMARK_STEP_SEC = 5.0


def median_state(distribution):
    return int(np.searchsorted(np.cumsum(distribution), 0.5, side="left"))


def load_case(case, adapter_row, hop):
    import partitura as pt

    performance_path = ROOT / case["files"]["midi.mid"]
    score_path = ROOT / case["files"]["score.mxl"]
    clock = np.load(ROOT / adapter_row["clock"])
    score_clock_sec = clock["score_clock_quarter"] * 0.5
    performance_clock_sec = clock["performance_clock_sec"]
    if np.any(np.diff(score_clock_sec) <= 0) or np.any(np.diff(performance_clock_sec) <= 0):
        raise AssertionError("adapted clock changed or is not strict")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        notes = pt.load_musicxml(str(score_path))[0].note_array()
    score_notes = [
        (
            float(row["onset_quarter"]) * 0.5,
            float(row["onset_quarter"] + row["duration_quarter"]) * 0.5,
            int(row["pitch"]),
        )
        for row in notes
    ]
    performance_notes = load_notes(performance_path)
    performance_duration = max(note[1] for note in performance_notes)
    score_duration = max(note[1] for note in score_notes)
    score_hop = hop * score_duration / performance_duration
    observations = notes_to_chroma_stream(performance_notes, performance_duration, hop)
    reference = notes_to_chroma_stream(score_notes, score_duration, score_hop)
    truth = np.interp(
        np.arange(len(observations)) * hop,
        performance_clock_sec,
        score_clock_sec,
        left=score_clock_sec[0],
        right=score_clock_sec[-1],
    ) / score_hop
    stop = min(performance_duration, float(performance_clock_sec[-1]))
    return observations, reference, score_hop, truth, stop


def main():
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf8"))
    adapter = json.loads(ADAPTER.read_text(encoding="utf8"))
    if not all(row["adapter_passed"] for row in adapter["rows"]):
        raise SystemExit("clock adapter gate failed")
    by_trial = {row["trial_id"]: row for row in adapter["rows"]}
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text(encoding="utf8"))
    hop = float(config["hop_sec"])
    weights = tuple(config["bayes"]["transition"])
    future = int(round(HORIZON_SEC / hop))
    window = int(round(WINDOW_SEC / hop))
    rows = []
    track_summaries = []
    for number, case in enumerate(acquisition["rows"], 1):
        trial = case["trial_id"]
        observations, reference, score_hop, truth, duration = load_case(case, by_trial[trial], hop)
        path, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        error = np.abs(path - truth) * score_hop
        local = []
        for time_sec in np.arange(15.0, duration - HORIZON_SEC - 0.1 + 1e-9, LANDMARK_STEP_SEC):
            end = int(round(time_sec / hop))
            endpoint = end + future
            if endpoint >= len(trace) or not healthy_before(error, end - 1, int(round(2 / hop)), 0.5):
                continue
            forecast = trace[end - 1].copy()
            for _ in range(future + 1):
                forecast = dc.apply_prediction(forecast, weights)
            center = median_state(forecast)
            covered = abs(center - truth[endpoint]) * score_hop <= RADIUS
            action = fp.projected_action(trace, weights, end - window, end, future + 1)
            low, high = tight.help_envelope(action, center, RADIUS, score_hop, CAP)
            certified = low > 0 or high < 0
            sign = 1 if low > 0 else -1 if high < 0 else 0
            states = np.arange(len(action))
            oracle = -float(np.minimum(np.abs(states - truth[endpoint]) * score_hop, CAP) @ action)
            record = {
                "trial_id": trial,
                "time_sec": float(time_sec),
                "covered": bool(covered),
                "certified": bool(certified),
                "sign": sign,
                "sign_match": bool(sign == np.sign(oracle)),
            }
            rows.append(record)
            local.append(record)
        track_summaries.append(
            {
                "trial_id": trial,
                "landmarks": len(local),
                "coverage": float(np.mean([row["covered"] for row in local])) if local else None,
                "certificate_share": float(np.mean([row["certified"] for row in local])) if local else None,
                "history_exclusion": case["history_exclusion"],
            }
        )
        print(f"[{number}/5] {trial}", flush=True)
    certificates = [row for row in rows if row["certified"]]
    covered_certificates = [row for row in certificates if row["covered"]]
    per_track_coverage = [row["coverage"] for row in track_summaries if row["coverage"] is not None]
    gates = {
        "five_samples": len(track_summaries) == 5,
        "landmarks_ge_25": len(rows) >= 25,
        "overall_coverage_ge_0_90": bool(rows and np.mean([row["covered"] for row in rows]) >= 0.90),
        "each_sample_coverage_ge_0_85": bool(per_track_coverage and min(per_track_coverage) >= 0.85),
        "certificate_share_ge_0_20": bool(rows and len(certificates) / len(rows) >= 0.20),
        "covered_sign_accuracy_ge_0_95": bool(
            covered_certificates and np.mean([row["sign_match"] for row in covered_certificates]) >= 0.95
        ),
    }
    result = {
        "scope": "frozen 12 s to 0.5 s position/sign feasibility on SKY-Piano public samples",
        "counts": {
            "samples": len(track_summaries),
            "landmarks": len(rows),
            "certificates": len(certificates),
            "covered_certificates": len(covered_certificates),
            "independent_long_pilot_works": sum(
                row["pilot_eligible"] for row in adapter["rows"]
            ),
        },
        "metrics": {
            "overall_coverage": float(np.mean([row["covered"] for row in rows])) if rows else None,
            "certificate_share": len(certificates) / len(rows) if rows else None,
            "covered_sign_accuracy": float(np.mean([row["sign_match"] for row in covered_certificates]))
            if covered_certificates
            else None,
        },
        "gates": gates,
        "qualified": bool(all(gates.values())),
        "track_summaries": track_summaries,
        "temporal_signal_computed": False,
        "natural_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "landmarks.json").write_text(json.dumps(rows), encoding="utf8")
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
