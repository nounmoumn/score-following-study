#!/usr/bin/env python3
"""One-shot P3-FHVB transfer to the five public SKY-Piano samples."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fhb_correction_burst_transfer as fast
import p3_fci_tight_envelope_diagnostic as tight
import p3_fp_forward_projection as fp
import p3_sky_piano_position_transfer as cohort
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

PROTOCOL = ROOT / "P3_SKY_PIANO_TEMPORAL_TRANSFER_PROTOCOL.md"
OUT = ROOT / "results/p3_sky_piano_temporal_transfer"
WINDOW_SEC = 12.0
HORIZON_SEC = 0.5
LAG_SEC = 0.5
STEP_SEC = 0.5
RADIUS = 0.5211881394162992
CAP = 5.0
SEED = 20260921
BOOTSTRAPS = 20_000
PERMUTATIONS = 2_000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median_state(distribution):
    return int(np.searchsorted(np.cumsum(distribution), 0.5, side="left"))


def summarize(rows, rng):
    events = [row["deficit"] for row in rows if row["event"]]
    nonevents = [row["deficit"] for row in rows if not row["event"]]
    tracks = {row["track"] for row in rows}
    event_tracks = {row["track"] for row in rows if row["event"]}
    onsets = {(row["track"], row["onset_sec"]) for row in rows if row["event"]}
    auc = fast.fast_auc(events, nonevents)
    if len(tracks) >= 2 and events and nonevents:
        interval = fast.fast_cluster_bootstrap(rows, rng, draws=BOOTSTRAPS)
        null = np.asarray([fast.fast_shift_auc(rows, rng) for _ in range(PERMUTATIONS)])
        null = null[np.isfinite(null)]
        p_value = float((1 + np.sum(null >= auc)) / (1 + len(null)))
    else:
        interval = [None, None]
        p_value = None
    return {
        "counts": {
            "samples": len(tracks),
            "cases": len(rows),
            "positive_cases": len(events),
            "unique_onsets": len(onsets),
            "event_samples": len(event_tracks),
        },
        "metrics": {
            "event_median": float(np.median(events)) if events else None,
            "nonevent_median": float(np.median(nonevents)) if nonevents else None,
            "auc": auc if np.isfinite(auc) else None,
            "sample_bootstrap_auc_95": interval,
            "circular_shift_p": p_value,
        },
    }


def main():
    position_result = json.loads((cohort.OUT / "result.json").read_text(encoding="utf8"))
    if not position_result["qualified"]:
        raise SystemExit("SKY-Piano position/sign transfer did not qualify")
    acquisition = json.loads(cohort.ACQUISITION.read_text(encoding="utf8"))
    adapter = json.loads(cohort.ADAPTER.read_text(encoding="utf8"))
    if len(acquisition["rows"]) != 5 or not all(row["adapter_passed"] for row in adapter["rows"]):
        raise SystemExit("sealed five-sample cohort changed")
    by_trial = {row["trial_id"]: row for row in adapter["rows"]}
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text(encoding="utf8"))
    hop = float(config["hop_sec"])
    weights = tuple(config["bayes"]["transition"])
    future = int(round(HORIZON_SEC / hop))
    window = int(round(WINDOW_SEC / hop))
    rows = []
    for number, case in enumerate(acquisition["rows"], 1):
        observations, reference, score_hop, truth, duration = cohort.load_case(
            case, by_trial[case["trial_id"]], hop
        )
        path, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        error = np.abs(path - truth) * score_hop
        raw = {}
        for time_sec in np.arange(max(WINDOW_SEC, 15.0), duration - HORIZON_SEC - 0.1 + 1e-9, STEP_SEC):
            end = int(round(time_sec / hop))
            endpoint = end + future
            if endpoint >= len(trace) or not healthy_before(error, end - 1, int(round(2 / hop)), 0.5):
                continue
            forecast = trace[end - 1].copy()
            for _ in range(future + 1):
                forecast = dc.apply_prediction(forecast, weights)
            center = median_state(forecast)
            action = fp.projected_action(trace, weights, end - window, end, future + 1)
            low, _ = tight.help_envelope(action, center, RADIUS, score_hop, CAP)
            onset = sustained_start(error, end, endpoint)
            raw[round(float(time_sec), 6)] = {
                "low": float(low),
                "event": bool(onset is not None),
                "onset_sec": None if onset is None else float(onset * hop),
            }
        for time_sec, record in raw.items():
            previous = raw.get(round(time_sec - LAG_SEC, 6))
            if previous is None:
                continue
            rows.append(
                {
                    "track": case["trial_id"],
                    "time_sec": time_sec,
                    "deficit": abs(record["low"] - previous["low"]),
                    "event": record["event"],
                    "onset_sec": record["onset_sec"],
                    "history_exclusion": case["history_exclusion"],
                }
            )
        print(f"[{number}/5] {case['trial_id']}", flush=True)
    rng = np.random.default_rng(SEED)
    full = summarize(rows, rng)
    independent_rows = [row for row in rows if not row["history_exclusion"]]
    sensitivity = summarize(independent_rows, rng)
    counts, metrics = full["counts"], full["metrics"]
    gates = {
        "five_samples": counts["samples"] == 5,
        "cases_ge_200": counts["cases"] >= 200,
        "events_ge_5_in_3_samples": counts["unique_onsets"] >= 5 and counts["event_samples"] >= 3,
        "event_median_gt_nonevent": bool(
            metrics["event_median"] is not None
            and metrics["nonevent_median"] is not None
            and metrics["event_median"] > metrics["nonevent_median"]
        ),
        "auc_ge_0_65": bool(metrics["auc"] is not None and metrics["auc"] >= 0.65),
        "bootstrap_lower_gt_0_55": bool(
            metrics["sample_bootstrap_auc_95"][0] is not None
            and metrics["sample_bootstrap_auc_95"][0] > 0.55
        ),
        "circular_shift_p_le_0_05": bool(
            metrics["circular_shift_p"] is not None and metrics["circular_shift_p"] <= 0.05
        ),
    }
    passed = all(gates.values())
    result = {
        "hypothesis": "frozen Batik P3-FHVB help-volatility precursor transfers to SKY-Piano public samples",
        **full,
        "gates": gates,
        "qualified": bool(passed),
        "sky_piano_cross_source_p3_prime_supported": bool(passed),
        "descriptive_no_prior_overlap_subset": sensitivity,
        "original_directional_natural_p3_confirmed": False,
        "limitations": [
            "public samples are selected rather than a complete release",
            "all five trials are from one professional recording session",
            "three works or families overlap prior ASAP-era research history",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "temporal_rows.json").write_text(json.dumps(rows), encoding="utf8")
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    provenance_files = [
        PROTOCOL,
        Path(__file__),
        cohort.ACQUISITION,
        cohort.ADAPTER,
        cohort.OUT / "result.json",
        ROOT / "configs/p3_temporal.yaml",
    ]
    (OUT / "provenance.json").write_text(
        json.dumps({str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in provenance_files}, indent=2),
        encoding="utf8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
