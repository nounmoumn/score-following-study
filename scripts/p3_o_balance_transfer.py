#!/usr/bin/env python3
"""One-shot frozen P3-O evaluation on 25 preflighted ASAP families."""

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
import p3_n_risk_balance as risk_balance
from p3_short_signal import sustained_start
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.ground_truth import piecewise_linear_gt
from score_following.temporal import healthy_before

MANIFEST = ROOT / "results/p3_onset_development/holdout_preflight/eligible_first_per_family.csv"
ANNOTATIONS = ROOT / "asap_dataset/asap_annotations.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_O_BALANCE_TRANSFER_PROTOCOL.md"
OUT = ROOT / "results/p3_o_balance_transfer"
LANDMARKS_SEC = tuple(range(15, 86, 5))
HISTORY = 20
FUTURE = 20
BOOTSTRAPS = 20000
PERMUTATIONS = 2000
SEED = 20260921


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rank_auc(positive, negative):
    if not positive or not negative:
        return float("nan")
    values = [1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative]
    return float(np.mean(values))


def aucs(rows):
    event = [row for row in rows if row["event"]]
    nonevent = [row for row in rows if not row["event"]]
    balance = rank_auc([row["balance_pressure"] for row in event],
                       [row["balance_pressure"] for row in nonevent])
    transport = rank_auc([row["transport_burden"] for row in event],
                         [row["transport_burden"] for row in nonevent])
    return balance, transport, balance - transport


def bootstrap(rows, rng):
    families = sorted({row["family"] for row in rows})
    grouped = {family: [row for row in rows if row["family"] == family] for family in families}
    values = []
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(families, len(families), replace=True)
        selected = [row for family in sample for row in grouped[family]]
        metrics = aucs(selected)
        if np.all(np.isfinite(metrics)):
            values.append(metrics)
    values = np.asarray(values)
    return ([float(np.percentile(values[:, 0], 2.5)), float(np.percentile(values[:, 0], 97.5))],
            [float(np.percentile(values[:, 2], 2.5)), float(np.percentile(values[:, 2], 97.5))])


def shifted_auc(rows, rng):
    shifted = []
    for family in sorted({row["family"] for row in rows}):
        group = [dict(row) for row in rows if row["family"] == family]
        if len(group) > 1:
            values = [row["balance_pressure"] for row in group]
            values = np.roll(values, int(rng.integers(1, len(group))))
            for row, value in zip(group, values):
                row["balance_pressure"] = float(value)
        shifted.extend(group)
    return aucs(shifted)[0]


def load_case(case, annotations, hop):
    base = ROOT / "asap_dataset"
    performance_notes = load_notes(base / case["performance"])
    score_notes = load_notes(base / case["score"])
    performance_duration = max(note[1] for note in performance_notes)
    score_duration = max(note[1] for note in score_notes)
    score_hop = hop * score_duration / performance_duration
    u = notes_to_chroma_stream(performance_notes, 90.0, hop)
    v = notes_to_chroma_stream(score_notes, score_duration, score_hop)
    annotation = annotations[case["performance"]]
    gt, _, _ = piecewise_linear_gt(annotation["performance_beats"], annotation["midi_score_beats"])
    truth = gt(np.arange(len(u)) * hop) / score_hop
    return u, v, score_hop, truth


def main():
    cases = read_csv(MANIFEST)
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    max_residual = 0.0
    certificate_violations = 0
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth = load_case(case, annotations, hop)
        cost = dose.cosine_cost(u, v)
        path, trace = dose.bayes_trace(cost, weights)
        frame_rows, risk, map_error = risk_balance.balance(trace, truth, score_hop, weights)
        by_frame = {row["frame"]: row for row in frame_rows}
        max_residual = max(max_residual, max(row["balance_residual"] for row in frame_rows))
        certificate_violations += sum(row["map_error"] > .5 and row["certificate_margin"] < -1e-12
                                      for row in frame_rows)
        for time_sec in LANDMARKS_SEC:
            end = int(round(time_sec / hop))
            if end + FUTURE + 4 >= len(map_error) or not healthy_before(map_error, end - 1, HISTORY, .5):
                continue
            segment = [by_frame[frame] for frame in range(end - HISTORY, end)]
            transport = float(sum(row["transport_burden"] for row in segment))
            correction = float(sum(row["observation_correction"] for row in segment))
            onset = sustained_start(map_error, end, end + FUTURE)
            rows.append({
                "family": case["work_family"], "composer": case["composer"],
                "time_sec": time_sec, "transport_burden": transport,
                "observation_correction": correction,
                "balance_pressure": transport - correction,
                "risk_change_check": float(risk[end - 1] - risk[end - HISTORY - 1]),
                "event": bool(onset is not None),
                "onset_sec": None if onset is None else float(onset * hop),
            })
        print(f"[{number}/{len(cases)}] {case['work_family']}")

    balance_auc, transport_auc, delta_auc = aucs(rows)
    rng = np.random.default_rng(SEED)
    auc_ci, delta_ci = bootstrap(rows, rng)
    null = np.asarray([shifted_auc(rows, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    shift_p = float((1 + np.sum(null >= balance_auc)) / (len(null) + 1))
    event = [row["balance_pressure"] for row in rows if row["event"]]
    nonevent = [row["balance_pressure"] for row in rows if not row["event"]]
    onsets = {(row["family"], row["onset_sec"]) for row in rows if row["event"]}
    event_families = {row["family"] for row in rows if row["event"]}
    coverage = (len(cases) == 25 and len(rows) >= 250 and len(onsets) >= 5
                and len(event_families) >= 4)
    gates = {
        "coverage": coverage,
        "event_median_gt_nonevent": bool(event and np.median(event) > np.median(nonevent)),
        "balance_auc_ge_0_70": balance_auc >= .70,
        "auc_bootstrap_lower_gt_0_55": auc_ci[0] > .55,
        "delta_auc_ge_0_05": delta_auc >= .05,
        "delta_auc_bootstrap_lower_gt_0": delta_ci[0] > 0,
        "circular_shift_p_le_0_05": shift_p <= .05,
        "balance_residual_le_1e_12": max_residual <= 1e-12,
        "zero_certificate_violations": certificate_violations == 0,
    }
    status = "pass" if all(gates.values()) else "information_limited" if not coverage else "negative"
    result = {
        "hypothesis": "P3-O natural balance pressure transfer",
        "scope": "one-shot oracle temporal mechanism on preserved ASAP Beethoven/Chopin families",
        "counts": {"families": len(cases), "eligible_landmarks": len(rows),
                   "unique_event_onsets": len(onsets), "event_families": len(event_families)},
        "metrics": {"event_balance_median": None if not event else float(np.median(event)),
                    "nonevent_balance_median": None if not nonevent else float(np.median(nonevent)),
                    "balance_auc": balance_auc, "transport_auc": transport_auc,
                    "delta_auc": delta_auc, "balance_auc_bootstrap_95": auc_ci,
                    "delta_auc_bootstrap_95": delta_ci, "circular_shift_p": shift_p,
                    "null_auc_p95": float(np.percentile(null, 95))},
        "audit": {"max_balance_residual": max_residual,
                  "certificate_violations": certificate_violations,
                  "max_risk_change_check_residual": float(max(abs(row["balance_pressure"] - row["risk_change_check"])
                                                               for row in rows))},
        "gates": {key: bool(value) for key, value in gates.items()},
        "status": status,
        "natural_temporal_p3_confirmed": status == "pass",
        "gt_free_warning_claim": False,
        "provenance": {"manifest_sha256": digest(MANIFEST), "annotations_sha256": digest(ANNOTATIONS),
                       "config_sha256": digest(CONFIG), "protocol_sha256": digest(PROTOCOL),
                       "script_sha256": digest(Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "landmarks.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

