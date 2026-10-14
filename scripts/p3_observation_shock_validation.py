#!/usr/bin/env python3
"""One-shot P3-OS validation on the preserved repaired-clock cohort."""
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
import p3_dc_conditional_erasure as dc
import p3_drift_cause_decomposition as causes
import p3_fci_tight_envelope_diagnostic as tight
import p3_fp_forward_projection as fp
import p3_observation_loss_selector as selector
from p3_fhb_correction_burst_transfer import fast_auc, fast_shift_auc
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_clock_repair_preflight/manifest.json"
OUT = ROOT / "results/p3_observation_shock_validation"
THRESHOLD = 0.20161929092323988
WINDOW_SEC = 12.0
HORIZON_SEC = 0.5
LAG_SEC = 0.5
STEP_SEC = 0.5
RADIUS = 0.5211881394162992
CAP = 5.0
PERMUTATIONS = 2_000
SEED = 20260921


def fixed_threshold_metrics(event_rows):
    actual = np.asarray([row["observation_loss"] for row in event_rows], dtype=bool)
    predicted = np.asarray([row["action_tv"] >= THRESHOLD for row in event_rows], dtype=bool)
    tp = int(np.sum(actual & predicted)); fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted)); fp_count = int(np.sum(~actual & predicted))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp_count) if tn + fp_count else float("nan")
    return {
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp_count},
    }


def main():
    import p3_fc_fresh_certificate as cohort

    manifest = json.loads(MANIFEST.read_text())
    cases = [row for row in manifest["rows"] if row["eligible"]]
    if len(cases) != 11:
        raise SystemExit("preserved 11-track manifest changed")
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text())
    hop = float(config["hop_sec"])
    weights = tuple(float(value) for value in config["bayes"]["transition"])
    future = int(round(HORIZON_SEC / hop)); window = int(round(WINDOW_SEC / hop))
    lag = int(round(LAG_SEC / hop)); history = int(round(2.0 / hop))
    temporal_rows, event_rows, residuals = [], [], []
    for number, case in enumerate(cases, 1):
        observations, reference, score_hop, truth, duration = cohort.load_case(case, hop)
        path, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        error = np.abs(path - truth) * score_hop
        decomposition = causes.decompose(trace, truth, score_hop, weights)
        by_frame = {row["frame"]: row for row in decomposition}
        residuals.extend((row["one_step_residual"], row["cumulative_residual"]) for row in decomposition)
        raw = {}
        for time_sec in np.arange(max(WINDOW_SEC, 15.0), duration - HORIZON_SEC - 0.1 + 1e-9, STEP_SEC):
            end = int(round(time_sec / hop)); endpoint = end + future
            if endpoint >= len(trace) or not healthy_before(error, end - 1, history, 0.5):
                continue
            forecast = trace[end - 1].copy()
            for _ in range(future + 1):
                forecast = dc.apply_prediction(forecast, weights)
            center = selector.median_state(forecast)
            action = fp.projected_action(trace, weights, end - window, end, future + 1)
            low, _ = tight.help_envelope(action, center, RADIUS, score_hop, CAP)
            onset = sustained_start(error, end, endpoint)
            features = selector.event_features(trace, end, score_hop, weights, hop)
            raw[round(float(time_sec), 6)] = {
                "end": end, "low": low, "event": onset is not None,
                "onset_sec": None if onset is None else float(onset * hop),
                "action_tv": -features["negative_action_tv"],
            }
        source_track = f"{case['source']}:{case['unit']}"
        seen_onsets = set()
        for time_sec, row in raw.items():
            old = raw.get(round(time_sec - LAG_SEC, 6))
            if old is None:
                continue
            temporal = {
                "track": source_track, "source": case["source"], "time_sec": time_sec,
                "deficit": abs(float(row["low"] - old["low"])),
                "event": bool(row["event"]), "onset_sec": row["onset_sec"],
                "action_tv": row["action_tv"],
            }
            temporal_rows.append(temporal)
            if not row["event"] or row["onset_sec"] in seen_onsets:
                continue
            seen_onsets.add(row["onset_sec"])
            onset_frame = int(round(row["onset_sec"] / hop))
            segment = [by_frame[frame] for frame in range(max(1, onset_frame - history), onset_frame) if frame in by_frame]
            sums, positive, label = causes.label_event(segment)
            event_rows.append({
                **temporal, "dominant_cause": label,
                "observation_loss": label == "observation_loss",
                **{f"{key}_sum": value for key, value in sums.items()},
                **{f"{key}_positive": value for key, value in positive.items()},
            })
        print(f"[{number}/{len(cases)}] {source_track}", flush=True)

    positive = [row["action_tv"] for row in event_rows if row["observation_loss"]]
    negative = [row["action_tv"] for row in event_rows if not row["observation_loss"]]
    shock_auc = fast_auc(positive, negative)
    threshold_metrics = fixed_threshold_metrics(event_rows)
    cause_lookup = {(row["track"], row["onset_sec"]): row["dominant_cause"] for row in event_rows}
    cause_temporal = []
    for row in temporal_rows:
        if row["event"] and cause_lookup.get((row["track"], row["onset_sec"])) != "observation_loss":
            continue
        cause_temporal.append(row)
    event_signal = [row["deficit"] for row in cause_temporal if row["event"]]
    nonevent_signal = [row["deficit"] for row in cause_temporal if not row["event"]]
    fhvb_auc = fast_auc(event_signal, nonevent_signal)
    rng = np.random.default_rng(SEED)
    null = np.asarray([fast_shift_auc(cause_temporal, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    shift_p = float((1 + np.sum(null >= fhvb_auc)) / (1 + len(null)))
    max_step = max(value[0] for value in residuals); max_cumulative = max(value[1] for value in residuals)
    observation_tracks = {row["track"] for row in event_rows if row["observation_loss"]}
    source_breakdown = {}
    for source in ("KRAISLER", "Batik"):
        source_events = [row for row in event_rows if row["source"] == source]
        source_positive = [row["action_tv"] for row in source_events if row["observation_loss"]]
        source_negative = [row["action_tv"] for row in source_events if not row["observation_loss"]]
        source_observation_keys = {
            (row["track"], row["onset_sec"]) for row in source_events if row["observation_loss"]
        }
        source_temporal = [
            row for row in temporal_rows if row["source"] == source
            and (not row["event"] or (row["track"], row["onset_sec"]) in source_observation_keys)
        ]
        source_event_signal = [row["deficit"] for row in source_temporal if row["event"]]
        source_nonevent_signal = [row["deficit"] for row in source_temporal if not row["event"]]
        source_breakdown[source] = {
            "onsets": len(source_events), "observation_loss_onsets": len(source_positive),
            "observation_loss_tracks": len({row["track"] for row in source_events if row["observation_loss"]}),
            "shock_auc": fast_auc(source_positive, source_negative) if source_positive and source_negative else None,
            "cause_specific_fhvb_auc": fast_auc(source_event_signal, source_nonevent_signal) if source_event_signal else None,
        }
    selected_temporal = [row for row in temporal_rows if row["action_tv"] >= THRESHOLD]
    selected_event = [row["deficit"] for row in selected_temporal if row["event"]]
    selected_nonevent = [row["deficit"] for row in selected_temporal if not row["event"]]
    selected_auc = fast_auc(selected_event, selected_nonevent)
    selected_rng = np.random.default_rng(SEED)
    selected_null = np.asarray([fast_shift_auc(selected_temporal, selected_rng) for _ in range(PERMUTATIONS)])
    selected_null = selected_null[np.isfinite(selected_null)]
    selected_p = float((1 + np.sum(selected_null >= selected_auc)) / (1 + len(selected_null)))
    gates = {
        "validation_outcomes_previously_unopened": False,
        "eleven_tracks": len(cases) == 11,
        "events_ge_10": len(event_rows) >= 10,
        "observation_loss_ge_4_in_3_tracks": len(positive) >= 4 and len(observation_tracks) >= 3,
        "decomposition_residual_le_1e_11": max(max_step, max_cumulative) <= 1e-11,
        "shock_auc_ge_0_70": shock_auc >= 0.70,
        "fixed_threshold_balanced_accuracy_ge_0_65": threshold_metrics["balanced_accuracy"] >= 0.65,
        "fhvb_event_median_gt_nonevent": bool(event_signal and np.median(event_signal) > np.median(nonevent_signal)),
        "fhvb_auc_ge_0_65": fhvb_auc >= 0.65,
        "fhvb_shift_p_le_0_05": shift_p <= 0.05,
    }
    result = {
        "hypothesis": "P3-OS observation shock selects observation-loss drift where FHVB precedes onset",
        "scope": "post-outcome diagnostic: all eleven repaired-clock outcomes were already opened",
        "provenance_audit": {
            "valid_independent_component": "none",
            "invalid_as_independent": "Batik six were opened by p3_fhvb6_final_validation.py; KRAISLER five by p3_fhvb_long_horizon_validation.py",
            "disposition": "retain all results, fail independence gate, do not claim P3-OS validation",
        },
        "counts": {
            "tracks": len(cases), "temporal_landmarks": len(temporal_rows), "unique_onsets": len(event_rows),
            "cause_counts": dict(Counter(row["dominant_cause"] for row in event_rows)),
            "observation_loss_tracks": len(observation_tracks),
        },
        "shock_selector": {"threshold": THRESHOLD, "auc": shock_auc, "fixed_threshold": threshold_metrics},
        "cause_specific_fhvb": {
            "event_median": float(np.median(event_signal)) if event_signal else None,
            "nonevent_median": float(np.median(nonevent_signal)), "auc": fhvb_auc,
            "within_track_shift_p": shift_p,
        },
        "source_breakdown": source_breakdown,
        "post_validation_operational_diagnostic": {
            "status": "not a pre-specified success gate",
            "fixed_threshold_selected_landmarks": len(selected_temporal),
            "selected_events": len(selected_event),
            "selected_event_tracks": len({row["track"] for row in selected_temporal if row["event"]}),
            "event_median": float(np.median(selected_event)),
            "nonevent_median": float(np.median(selected_nonevent)),
            "auc": selected_auc,
            "within_track_shift_p": selected_p,
        },
        "max_decomposition_residual": {"one_step": max_step, "cumulative": max_cumulative},
        "gates": {key: bool(value) for key, value in gates.items()},
        "conditional_p3_prime_confirmed": bool(all(gates.values())),
        "original_directional_corpus_general_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("temporal_rows.csv", temporal_rows), ("events.csv", event_rows)):
        with (OUT / filename).open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
