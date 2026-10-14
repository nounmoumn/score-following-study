#!/usr/bin/env python3
"""One-shot Vienna temporal-endpoint validation of P3-OS."""
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
import p3_observation_shock_validation as shock
import p3_vienna_forecast_validation as vienna
from p3_fhb_correction_burst_transfer import fast_auc, fast_cluster_bootstrap, fast_shift_auc
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_vienna_forecast_preflight/manifest.json"
PROVENANCE = ROOT / "results/p3_vienna_temporal_provenance_audit/result.json"
OUT = ROOT / "results/p3_vienna_observation_shock_validation"
WINDOW_SEC = 12.0
HORIZON_SEC = 0.5
LAG_SEC = 0.5
STEP_SEC = 0.5
RADIUS = 0.5211881394162992
CAP = 5.0
PERMUTATIONS = 2_000
SEED = 20260921


def work_summary(work, events, temporal):
    work_events = [row for row in events if row["work"] == work]
    keys = {(row["track"], row["onset_sec"]) for row in work_events if row["observation_loss"]}
    selected = [
        row for row in temporal if row["work"] == work
        and (not row["event"] or (row["track"], row["onset_sec"]) in keys)
    ]
    event_signal = [row["deficit"] for row in selected if row["event"]]
    nonevent_signal = [row["deficit"] for row in selected if not row["event"]]
    return {
        "onsets": len(work_events),
        "observation_loss_onsets": len(keys),
        "observation_loss_performances": len({track for track, _ in keys}),
        "cause_specific_fhvb_auc": fast_auc(event_signal, nonevent_signal) if event_signal else None,
    }


def main():
    provenance = json.loads(PROVENANCE.read_text())
    if not provenance["no_prior_natural_temporal_endpoint_code"]:
        raise SystemExit("Vienna temporal endpoint was not preserved")
    manifest = json.loads(MANIFEST.read_text())
    cases = [row for row in manifest["rows"] if row["split"] == "validation"]
    if len(cases) != 44 or {row["work"] for row in cases} != {"Chopin_op38", "Schubert_D783_no15"}:
        raise SystemExit("frozen Vienna validation split changed")
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text())
    hop = float(config["hop_sec"]); weights = tuple(float(x) for x in config["bayes"]["transition"])
    future = int(round(HORIZON_SEC / hop)); window = int(round(WINDOW_SEC / hop))
    lag = int(round(LAG_SEC / hop)); history = int(round(2.0 / hop))
    temporal_rows, event_rows, residuals = [], [], []
    cache = {}
    for number, case in enumerate(cases, 1):
        observations, reference, score_hop, truth, duration = vienna.load_case(case, hop, cache)
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
        seen_onsets = set()
        for time_sec, row in raw.items():
            old = raw.get(round(time_sec - LAG_SEC, 6))
            if old is None:
                continue
            temporal = {
                "track": case["stem"], "work": case["work"], "time_sec": time_sec,
                "deficit": abs(float(row["low"] - old["low"])), "event": bool(row["event"]),
                "onset_sec": row["onset_sec"], "action_tv": row["action_tv"],
            }
            temporal_rows.append(temporal)
            if not row["event"] or row["onset_sec"] in seen_onsets:
                continue
            seen_onsets.add(row["onset_sec"])
            onset_frame = int(round(row["onset_sec"] / hop))
            segment = [by_frame[frame] for frame in range(max(1, onset_frame - history), onset_frame) if frame in by_frame]
            sums, positive, label = causes.label_event(segment)
            event_rows.append({
                **temporal, "dominant_cause": label, "observation_loss": label == "observation_loss",
                **{f"{key}_sum": value for key, value in sums.items()},
                **{f"{key}_positive": value for key, value in positive.items()},
            })
        print(f"[{number}/{len(cases)}] {case['stem']}", flush=True)

    positive = [row["action_tv"] for row in event_rows if row["observation_loss"]]
    negative = [row["action_tv"] for row in event_rows if not row["observation_loss"]]
    shock_auc = fast_auc(positive, negative)
    threshold_metrics = shock.fixed_threshold_metrics(event_rows)
    observation_keys = {(row["track"], row["onset_sec"]) for row in event_rows if row["observation_loss"]}
    cause_temporal = [
        row for row in temporal_rows
        if not row["event"] or (row["track"], row["onset_sec"]) in observation_keys
    ]
    event_signal = [row["deficit"] for row in cause_temporal if row["event"]]
    nonevent_signal = [row["deficit"] for row in cause_temporal if not row["event"]]
    fhvb_auc = fast_auc(event_signal, nonevent_signal)
    rng = np.random.default_rng(SEED)
    null = np.asarray([fast_shift_auc(cause_temporal, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    shift_p = float((1 + np.sum(null >= fhvb_auc)) / (1 + len(null)))
    bootstrap_ci = fast_cluster_bootstrap(cause_temporal, np.random.default_rng(SEED), draws=20_000)
    max_step = max(value[0] for value in residuals); max_cumulative = max(value[1] for value in residuals)
    works = {row["work"] for row in cases}
    work_results = {work: work_summary(work, event_rows, temporal_rows) for work in sorted(works)}
    observation_tracks = {row["track"] for row in event_rows if row["observation_loss"]}
    gates = {
        "temporal_endpoint_previously_unopened": True,
        "forty_four_hash_checked_performances": len(cases) == 44,
        "events_ge_10": len(event_rows) >= 10,
        "observation_loss_ge_4_in_3_performances": len(positive) >= 4 and len(observation_tracks) >= 3,
        "observation_loss_in_both_works": all(work_results[work]["observation_loss_onsets"] >= 1 for work in works),
        "decomposition_residual_le_1e_11": max(max_step, max_cumulative) <= 1e-11,
        "shock_auc_ge_0_70": shock_auc >= 0.70,
        "fixed_threshold_balanced_accuracy_ge_0_65": threshold_metrics["balanced_accuracy"] >= 0.65,
        "fhvb_event_median_gt_nonevent": bool(event_signal and np.median(event_signal) > np.median(nonevent_signal)),
        "fhvb_auc_ge_0_65": fhvb_auc >= 0.65,
        "fhvb_shift_p_le_0_05": shift_p <= 0.05,
        "each_work_fhvb_auc_ge_0_60": all(
            work_results[work]["cause_specific_fhvb_auc"] is not None
            and work_results[work]["cause_specific_fhvb_auc"] >= 0.60 for work in works
        ),
    }
    result = {
        "hypothesis": "P3-OS observation shock selects observation-loss drift where FHVB precedes onset",
        "scope": "one-shot Vienna validation-split natural temporal endpoint",
        "counts": {
            "performances": len(cases), "works": len(works), "temporal_landmarks": len(temporal_rows),
            "unique_onsets": len(event_rows), "cause_counts": dict(Counter(row["dominant_cause"] for row in event_rows)),
            "observation_loss_performances": len(observation_tracks),
        },
        "shock_selector": {"threshold": shock.THRESHOLD, "auc": shock_auc, "fixed_threshold": threshold_metrics},
        "cause_specific_fhvb": {
            "event_median": float(np.median(event_signal)) if event_signal else None,
            "nonevent_median": float(np.median(nonevent_signal)), "auc": fhvb_auc,
            "performance_bootstrap_auc_95": bootstrap_ci, "performance_shift_p": shift_p,
        },
        "work_results": work_results,
        "max_decomposition_residual": {"one_step": max_step, "cumulative": max_cumulative},
        "gates": {key: bool(value) for key, value in gates.items()},
        "vienna_p3_os_confirmed": bool(all(gates.values())),
        "original_directional_corpus_general_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("temporal_rows.csv", temporal_rows), ("events.csv", event_rows)):
        with (OUT / filename).open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
