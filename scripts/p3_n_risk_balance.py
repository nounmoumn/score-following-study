#!/usr/bin/env python3
"""Audit the exact P3-N posterior-risk balance on natural KRAISLER data."""

from __future__ import annotations

import csv
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
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_N_RISK_BALANCE_PROTOCOL.md"
OUT = ROOT / "results/p3_n_risk_balance"
CAP_SEC = 5.0
THRESHOLD_SEC = 0.5
HISTORY = 20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictive_prior(posterior, weights, restart=dose.RESTART):
    prior = np.zeros_like(posterior)
    for step, weight in enumerate(weights):
        if step == 0:
            prior += weight * posterior
        else:
            prior[step:] += weight * posterior[:-step]
            prior[-1] += weight * posterior[-step:].sum()
    return (1.0 - restart) * prior + restart / len(posterior)


def balance(trace, truth, score_hop, weights, cap=CAP_SEC):
    states = np.arange(trace.shape[1])
    risk = np.empty(len(trace))
    pmax = trace.max(axis=1)
    map_error = np.abs(trace.argmax(axis=1) - truth) * score_hop
    for frame in range(len(trace)):
        distance = np.minimum(np.abs(states - truth[frame]) * score_hop, cap)
        risk[frame] = trace[frame] @ distance
    rows = []
    cumulative = 0.0
    for frame in range(1, len(trace)):
        prior = predictive_prior(trace[frame - 1], weights)
        distance = np.minimum(np.abs(states - truth[frame]) * score_hop, cap)
        prior_risk = float(prior @ distance)
        transport = prior_risk - risk[frame - 1]
        correction = prior_risk - risk[frame]
        increment = risk[frame] - risk[frame - 1]
        cumulative += transport - correction
        rows.append({
            "frame": frame,
            "risk": float(risk[frame]),
            "transport_burden": float(transport),
            "observation_correction": float(correction),
            "risk_increment": float(increment),
            "balance_residual": float(abs(increment - (transport - correction))),
            "cumulative_residual": float(abs((risk[frame] - risk[0]) - cumulative)),
            "map_error": float(map_error[frame]),
            "pmax": float(pmax[frame]),
            "certificate_margin": float(risk[frame] - pmax[frame] * THRESHOLD_SEC),
        })
    return rows, risk, map_error


def main():
    from p3_kraisler_signal_run import load_case
    from p3_batik_external_run import load_gt

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    audits = []
    events = []
    path_mismatches = 0
    for number, case in enumerate(cases, 1):
        u, v, score_hop, _ = load_case(case, hop)
        beats, perf_times = load_gt(ROOT / case["match"])
        truth = np.interp(np.arange(len(u)) * hop, perf_times, beats * .5) / score_hop
        cost = dose.cosine_cost(u, v)
        path, trace = dose.bayes_trace(cost, weights)
        path_mismatches += int(np.sum(path != trace.argmax(axis=1)))
        rows, risk, map_error = balance(trace, truth, score_hop, weights)
        for row in rows:
            row["track"] = case["track"]
        audits.extend(rows)

        seen = set()
        candidates = [int(round(t / hop)) for t in np.arange(15., len(u) * hop - 5. + 1e-9, 5.)]
        for end in candidates:
            if (end + 20 >= len(map_error)
                    or not healthy_before(map_error, end - 1, HISTORY, THRESHOLD_SEC)):
                continue
            onset = sustained_start(map_error, end, end + 20)
            if onset is None or onset in seen:
                continue
            seen.add(onset)
            lo = max(1, onset - HISTORY)
            segment = [row for row in rows if lo <= row["frame"] < onset]
            events.append({
                "track": case["track"], "onset_sec": float(onset * hop),
                "pre_frames": len(segment),
                "transport_sum": float(sum(row["transport_burden"] for row in segment)),
                "correction_sum": float(sum(row["observation_correction"] for row in segment)),
                "net_risk_change_sum": float(sum(row["risk_increment"] for row in segment)),
                "observed_risk_change": float(risk[onset - 1] - risk[lo - 1]),
            })
        print(f"[{number}/{len(cases)}] track={case['track']}")

    max_step = max(row["balance_residual"] for row in audits)
    max_cumulative = max(row["cumulative_residual"] for row in audits)
    violations = sum(row["map_error"] > THRESHOLD_SEC and row["certificate_margin"] < -1e-12
                     for row in audits)
    gates = {
        "coverage_13_tracks": len(cases) == 13,
        "one_step_residual_le_1e_12": max_step <= 1e-12,
        "cumulative_residual_le_1e_11": max_cumulative <= 1e-11,
        "zero_map_certificate_violations": violations == 0,
        "zero_tracker_path_mismatches": path_mismatches == 0,
    }
    result = {
        "hypothesis": "P3-N exact natural-risk balance and MAP certificate",
        "scope": "exact mechanistic accounting; event attribution descriptive; no forecast claim",
        "counts": {"tracks": len(cases), "audited_frames": len(audits),
                   "descriptive_unique_events": len(events)},
        "parameters": {"cap_score_sec": CAP_SEC, "map_threshold_score_sec": THRESHOLD_SEC,
                       "descriptive_history_sec": HISTORY * hop},
        "max_residuals": {"one_step": max_step, "cumulative": max_cumulative},
        "certificate_violations": violations,
        "path_mismatches": path_mismatches,
        "event_descriptive": {
            "mean_transport_sum": float(np.mean([row["transport_sum"] for row in events])),
            "mean_correction_sum": float(np.mean([row["correction_sum"] for row in events])),
            "mean_net_risk_change": float(np.mean([row["observed_risk_change"] for row in events])),
            "events_with_positive_correction": int(sum(row["correction_sum"] > 0 for row in events)),
            "events_with_positive_risk_change": int(sum(row["observed_risk_change"] > 0 for row in events)),
        },
        "gates": {key: bool(value) for key, value in gates.items()},
        "identity_confirmed": bool(all(gates.values())),
        "natural_temporal_p3_confirmed": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("frames.csv", audits), ("events.csv", events)):
        with (OUT / name).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
