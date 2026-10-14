#!/usr/bin/env python3
"""One-shot long-horizon P3-FHVB validation on sealed KRAISLER-five."""
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
import p3_fhlr_batik_repair_transfer as repaired
import p3_fp_forward_projection as fp
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_clock_repair_preflight/manifest.json"
PROTOCOL = ROOT / "P3_FHVB_LONG_HORIZON_PROTOCOL.md"
OUT = ROOT / "results/p3_fhvb_long_horizon_validation"
WINDOW_SEC = 12.0
PROJECTION_HORIZON_SEC = 0.5
EVENT_HORIZON_SEC = 1.5
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


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"] and row["source"] == "KRAISLER"]
    if len(cases) != 5 or {row["unit"] for row in cases} != {"02", "03", "06", "09", "17"}:
        raise SystemExit("sealed KRAISLER-five cohort changed")
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text(encoding="utf8"))
    hop = float(config["hop_sec"])
    weights = tuple(config["bayes"]["transition"])
    projection = int(round(PROJECTION_HORIZON_SEC / hop))
    event_horizon = int(round(EVENT_HORIZON_SEC / hop))
    window = int(round(WINDOW_SEC / hop))
    rows = []
    for number, case in enumerate(cases, 1):
        observations, reference, score_hop, truth, duration = repaired.repaired_load(case, hop)
        path, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        error = np.abs(path - truth) * score_hop
        raw = {}
        for time_sec in np.arange(max(WINDOW_SEC, 15.0), duration - EVENT_HORIZON_SEC - 0.1 + 1e-9, STEP_SEC):
            end = int(round(time_sec / hop))
            projected_endpoint = end + projection
            event_endpoint = end + event_horizon
            if event_endpoint >= len(trace) or not healthy_before(error, end - 1, int(round(2 / hop)), 0.5):
                continue
            forecast = trace[end - 1].copy()
            for _ in range(projection + 1):
                forecast = dc.apply_prediction(forecast, weights)
            center = median_state(forecast)
            action = fp.projected_action(trace, weights, end - window, end, projection + 1)
            low, _ = tight.help_envelope(action, center, RADIUS, score_hop, CAP)
            onset = sustained_start(error, end, event_endpoint)
            raw[round(float(time_sec), 6)] = {
                "low": float(low),
                "event": bool(onset is not None),
                "onset_sec": None if onset is None else float(onset * hop),
            }
        for time_sec, record in raw.items():
            previous = raw.get(round(time_sec - LAG_SEC, 6))
            if previous is not None:
                rows.append(
                    {
                        "track": case["unit"],
                        "time_sec": time_sec,
                        "deficit": abs(record["low"] - previous["low"]),
                        "event": record["event"],
                        "onset_sec": record["onset_sec"],
                    }
                )
        print(f"[{number}/5] KRAISLER-{case['unit']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "temporal_rows.json").write_text(json.dumps(rows), encoding="utf8")
    events = [row["deficit"] for row in rows if row["event"]]
    nonevents = [row["deficit"] for row in rows if not row["event"]]
    auc = fast.fast_auc(events, nonevents)
    rng = np.random.default_rng(SEED)
    interval = fast.fast_cluster_bootstrap(rows, rng, draws=BOOTSTRAPS)
    null = np.asarray([fast.fast_shift_auc(rows, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    p_value = float((1 + np.sum(null >= auc)) / (1 + len(null)))
    tracks = {row["track"] for row in rows}
    event_tracks = {row["track"] for row in rows if row["event"]}
    onsets = {(row["track"], row["onset_sec"]) for row in rows if row["event"]}
    gates = {
        "five_tracks": len(tracks) == 5,
        "cases_ge_200": len(rows) >= 200,
        "events_ge_5_in_3_tracks": len(onsets) >= 5 and len(event_tracks) >= 3,
        "event_median_gt_nonevent": bool(events and np.median(events) > np.median(nonevents)),
        "auc_ge_0_65": bool(auc >= 0.65),
        "bootstrap_lower_gt_0_55": bool(interval[0] > 0.55),
        "circular_shift_p_le_0_05": bool(p_value <= 0.05),
    }
    passed = all(gates.values())
    result = {
        "hypothesis": "unchanged P3-FHVB signal precedes natural drift within 1.5 seconds",
        "counts": {
            "tracks": len(tracks),
            "cases": len(rows),
            "positive_cases": len(events),
            "unique_onsets": len(onsets),
            "event_tracks": len(event_tracks),
        },
        "metrics": {
            "event_median": float(np.median(events)) if events else None,
            "nonevent_median": float(np.median(nonevents)) if nonevents else None,
            "auc": auc,
            "track_bootstrap_auc_95": interval,
            "circular_shift_p": p_value,
        },
        "gates": gates,
        "validated": bool(passed),
        "kraisler_long_horizon_p3_prime_confirmed": bool(passed),
        "original_directional_natural_p3_confirmed": False,
        "selection_history": "1.5-second event horizon selected from exploratory SKY-Piano lead-lag profile",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    provenance_files = [PROTOCOL, Path(__file__), MANIFEST, ROOT / "configs/p3_temporal.yaml"]
    (OUT / "provenance.json").write_text(
        json.dumps({str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in provenance_files}, indent=2),
        encoding="utf8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
