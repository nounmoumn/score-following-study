#!/usr/bin/env python3
"""Exact transition/occurrence/observation decomposition of drift onsets."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import p3_d_dose_response as dose
import p3_n_risk_balance as risk_balance

OUT = ROOT / "results/p3_drift_cause_decomposition"
MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
EVENT_ROWS = ROOT / "results/p3_fhv_help_volatility_development/development_rows.json"
HISTORY_FRAMES = 20
CAP_SEC = 5.0


def decompose(trace, truth, score_hop, weights, cap=CAP_SEC):
    states = np.arange(trace.shape[1])
    rows = []
    cumulative_terms = 0.0
    initial_distance = np.minimum(np.abs(states - truth[0]) * score_hop, cap)
    initial_risk = float(trace[0] @ initial_distance)
    previous_risk = initial_risk
    for frame in range(1, len(trace)):
        prior = risk_balance.predictive_prior(trace[frame - 1], weights)
        previous_distance = np.minimum(np.abs(states - truth[frame - 1]) * score_hop, cap)
        current_distance = np.minimum(np.abs(states - truth[frame]) * score_hop, cap)
        previous_same_truth = float(trace[frame - 1] @ previous_distance)
        prior_previous_truth = float(prior @ previous_distance)
        prior_current_truth = float(prior @ current_distance)
        current_risk = float(trace[frame] @ current_distance)
        transition = prior_previous_truth - previous_same_truth
        occurrence = prior_current_truth - prior_previous_truth
        correction = prior_current_truth - current_risk
        increment = current_risk - previous_risk
        reconstructed = transition + occurrence - correction
        cumulative_terms += reconstructed
        rows.append(
            {
                "frame": frame,
                "risk": current_risk,
                "risk_increment": increment,
                "transition_burden": transition,
                "occurrence_burden": occurrence,
                "observation_correction": correction,
                "one_step_residual": abs(increment - reconstructed),
                "cumulative_residual": abs((current_risk - initial_risk) - cumulative_terms),
            }
        )
        previous_risk = current_risk
    return rows


def label_event(segment):
    sums = {
        "transition": float(sum(row["transition_burden"] for row in segment)),
        "occurrence": float(sum(row["occurrence_burden"] for row in segment)),
        "observation_loss": float(-sum(row["observation_correction"] for row in segment)),
    }
    positive = {key: max(value, 0.0) for key, value in sums.items()}
    total = sum(positive.values())
    if total <= 0:
        label = "no_positive_burden"
    else:
        winner = max(positive, key=positive.get)
        label = winner if positive[winner] / total >= 0.5 else "mixed"
    return sums, positive, label


def main():
    from p3_batik_external_run import load_gt
    from p3_kraisler_signal_run import load_case

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    event_data = json.loads(EVENT_ROWS.read_text(encoding="utf8"))["KRAISLER"]
    onsets = {}
    for row in event_data:
        if row["event"]:
            onsets.setdefault(row["track"], set()).add(float(row["onset_sec"]))
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text(encoding="utf8"))
    hop = float(config["hop_sec"])
    weights = tuple(config["bayes"]["transition"])
    all_frames = []
    events = []
    for number, case in enumerate(cases, 1):
        observations, reference, score_hop, _ = load_case(case, hop)
        score_clock, performance_clock = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(observations)) * hop, performance_clock, score_clock * 0.5) / score_hop
        _, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        frames = decompose(trace, truth, score_hop, weights)
        for row in frames:
            row["track"] = case["track"]
        all_frames.extend(frames)
        by_frame = {row["frame"]: row for row in frames}
        for onset_sec in sorted(onsets.get(case["track"], set())):
            onset = int(round(onset_sec / hop))
            segment = [by_frame[frame] for frame in range(max(1, onset - HISTORY_FRAMES), onset) if frame in by_frame]
            sums, positive, label = label_event(segment)
            events.append(
                {
                    "track": case["track"],
                    "onset_sec": onset_sec,
                    "pre_frames": len(segment),
                    **{f"{key}_sum": value for key, value in sums.items()},
                    **{f"{key}_positive": value for key, value in positive.items()},
                    "dominant_cause": label,
                }
            )
        print(f"[{number}/{len(cases)}] KRAISLER-{case['track']}", flush=True)
    max_step = max(row["one_step_residual"] for row in all_frames)
    max_cumulative = max(row["cumulative_residual"] for row in all_frames)
    gates = {
        "thirteen_tracks": len(cases) == 13,
        "events_match_opened_rows": len(events) == len({(row["track"], row["onset_sec"]) for row in event_data if row["event"]}),
        "one_step_residual_le_1e_12": max_step <= 1e-12,
        "cumulative_residual_le_1e_11": max_cumulative <= 1e-11,
        "all_event_windows_complete": all(row["pre_frames"] == HISTORY_FRAMES for row in events),
    }
    result = {
        "scope": "retrospective exact cause labels on opened KRAISLER development onsets",
        "identity": "risk increment = transition burden + occurrence burden - observation correction",
        "counts": {"tracks": len(cases), "frames": len(all_frames), "events": len(events)},
        "cause_counts": dict(Counter(row["dominant_cause"] for row in events)),
        "max_residuals": {"one_step": max_step, "cumulative": max_cumulative},
        "gates": gates,
        "qualified_for_selector_development": bool(all(gates.values()) and len(events) >= 10),
        "natural_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("frames.csv", all_frames), ("events.csv", events)):
        with (OUT / filename).open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
