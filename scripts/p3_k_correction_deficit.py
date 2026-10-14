#!/usr/bin/env python3
"""Frozen KRAISLER test of P3-J-derived natural correction deficit."""

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
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_kraisler_signal/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_K_CORRECTION_DEFICIT_PROTOCOL.md"
OUT = ROOT / "results/p3_k_correction_deficit"
HISTORY = 20
FUTURE = 20
CAP_SEC = 5.0
BOOTSTRAPS = 20000
PERMUTATIONS = 2000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_auc_safe(event, nonevent):
    if not event or not nonevent:
        return float("nan")
    comparisons = [1.0 if positive > negative else 0.5 if positive == negative else 0.0
                   for positive in event for negative in nonevent]
    return float(np.mean(comparisons))


def cluster_bootstrap_auc(rows, rng):
    tracks = sorted({row["track"] for row in rows})
    grouped = {track: [row for row in rows if row["track"] == track] for track in tracks}
    values = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(tracks, len(tracks), replace=True)
        selected = [row for track in sampled for row in grouped[track]]
        values[draw] = rank_auc_safe([r["deficit"] for r in selected if r["event"]],
                                     [r["deficit"] for r in selected if not r["event"]])
    values = values[np.isfinite(values)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def shifted_auc(rows, rng):
    shifted = []
    for track in sorted({row["track"] for row in rows}):
        group = [dict(row) for row in rows if row["track"] == track]
        if len(group) > 1:
            offset = int(rng.integers(1, len(group)))
            values = np.roll([row["deficit"] for row in group], offset)
            for row, value in zip(group, values):
                row["deficit"] = float(value)
        shifted.extend(group)
    return rank_auc_safe([r["deficit"] for r in shifted if r["event"]],
                         [r["deficit"] for r in shifted if not r["event"]])


def main():
    from p3_kraisler_signal_run import load_case
    from p3_batik_external_run import load_gt

    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(x) for x in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, duration = load_case(case, hop)
        beats, perf_times = load_gt(ROOT / case["match"])
        times = np.arange(len(u)) * hop
        truth = np.interp(times, perf_times, beats * .5) / score_hop
        cost = dose.cosine_cost(u, v)
        path, trace = dose.bayes_trace(cost, weights)
        error = np.abs(path - truth) * score_hop

        priors = np.empty_like(trace)
        priors[0] = np.eye(1, len(v), 0)[0]
        for frame in range(1, len(trace)):
            prior = np.zeros(len(v))
            for step, weight in enumerate(weights):
                if step == 0:
                    prior += weight * trace[frame - 1]
                else:
                    prior[step:] += weight * trace[frame - 1, :-step]
                    prior[-1] += weight * trace[frame - 1, -step:].sum()
            priors[frame] = (1 - dose.RESTART) * prior + dose.RESTART / len(v)

        for time_sec in np.arange(15.0, duration - 5.0 + 1e-9, 5.0):
            end = int(round(time_sec / hop))
            if end + FUTURE >= len(error) or not healthy_before(error, end - 1, HISTORY, .5):
                continue
            benefits = []
            actions = []
            states = np.arange(len(v))
            for frame in range(end - HISTORY, end):
                distance = np.minimum(np.abs(states - truth[frame]) * score_hop, CAP_SEC)
                benefits.append(float(np.sum((priors[frame] - trace[frame]) * distance)))
                actions.append(float(0.5 * np.abs(trace[frame] - priors[frame]).sum()))
            onset = sustained_start(error, end, end + FUTURE)
            rows.append({
                "track": case["track"], "time_sec": float(time_sec),
                "deficit": float(-np.mean(benefits)),
                "mean_correction_benefit": float(np.mean(benefits)),
                "mean_action_total_variation": float(np.mean(actions)),
                "event": bool(onset is not None),
                "onset_sec": None if onset is None else float(onset * hop),
            })
        print(f"[{number}/{len(cases)}] track={case['track']}")

    event = [row["deficit"] for row in rows if row["event"]]
    nonevent = [row["deficit"] for row in rows if not row["event"]]
    auc = rank_auc_safe(event, nonevent)
    rng = np.random.default_rng(SEED)
    auc_ci = cluster_bootstrap_auc(rows, rng)
    null = np.asarray([shifted_auc(rows, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    shift_p = float((1 + np.sum(null >= auc)) / (len(null) + 1))
    event_onsets = {(row["track"], row["onset_sec"]) for row in rows if row["event"]}
    event_tracks = {row["track"] for row in rows if row["event"]}
    metrics = {
        "event_deficit_median": float(np.median(event)),
        "nonevent_deficit_median": float(np.median(nonevent)),
        "landmark_rank_auc": auc,
        "track_bootstrap_auc_95": auc_ci,
        "circular_shift_p": shift_p,
        "null_auc_p95": float(np.percentile(null, 95)),
    }
    gates = {
        "coverage": len(cases) == 13 and len(rows) >= 100 and len(event_onsets) >= 5 and len(event_tracks) >= 5,
        "event_median_gt_nonevent": np.median(event) > np.median(nonevent),
        "auc_ge_0_70": auc >= .70,
        "bootstrap_lower_gt_0_55": auc_ci[0] > .55,
        "circular_shift_p_le_0_05": shift_p <= .05,
    }
    result = {
        "hypothesis": "P3-K natural correction-deficit temporal precursor",
        "scope": "KRAISLER oracle mechanism development; future event used only as outcome",
        "counts": {"tracks": len(cases), "landmarks": len(rows),
                   "unique_event_onsets": len(event_onsets), "event_tracks": len(event_tracks)},
        "parameters": {"history_sec": HISTORY * hop, "future_sec": FUTURE * hop,
                       "distance_cap_score_sec": CAP_SEC, "seed": SEED,
                       "bootstraps": BOOTSTRAPS, "permutations": PERMUTATIONS},
        "metrics": metrics,
        "gates": {key: bool(value) for key, value in gates.items()},
        "development_pass": bool(all(gates.values())),
        "cross_corpus_authorized": bool(all(gates.values())),
        "deployable_gt_free_claim": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "config_sha256": digest(CONFIG),
                       "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))},
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
