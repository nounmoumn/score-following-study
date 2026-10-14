#!/usr/bin/env python3
"""Past-only development selector for observation-loss drift events."""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fci_tight_envelope_diagnostic as tight
import p3_fp_forward_projection as fp
from p3_fhb_correction_burst_transfer import fast_auc

OUT = ROOT / "results/p3_observation_loss_selector"
WINDOW_SEC = 12.0
HORIZON_SEC = 0.5
ACTION_SEC = 2.0
RADIUS = 0.5211881394162992
CAP = 5.0


def median_state(p):
    return int(np.searchsorted(np.cumsum(p), 0.5, side="left"))


def event_features(trace, end, score_hop, weights, hop):
    """Features available strictly before landmark ``end``."""
    future = int(round(HORIZON_SEC / hop))
    window = int(round(WINDOW_SEC / hop))
    action_frames = int(round(ACTION_SEC / hop))
    forecast = trace[end - 1].copy()
    for _ in range(future + 1):
        forecast = dc.apply_prediction(forecast, weights)
    center = median_state(forecast)
    z = fp.projected_action(trace, weights, end - window, end, future + 1)
    low, high = tight.help_envelope(z, center, RADIUS, score_hop, CAP)
    posterior = trace[end - 1]
    positive = posterior[posterior > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(len(posterior))
    tv = []
    for frame in range(end - action_frames, end):
        prior = dc.apply_prediction(trace[frame - 1], weights)
        tv.append(0.5 * float(np.abs(trace[frame] - prior).sum()))
    return {
        "help_lower": low,
        "help_upper": high,
        "help_deficit": -low,
        "posterior_entropy": entropy,
        "negative_pmax": -float(posterior.max()),
        "negative_action_tv": -float(np.mean(tv)),
    }


def loto_midpoint(rows, key):
    predictions = []
    for heldout in sorted({row["track"] for row in rows}):
        train = [row for row in rows if row["track"] != heldout]
        pos = [row[key] for row in train if row["observation_loss"]]
        neg = [row[key] for row in train if not row["observation_loss"]]
        threshold = 0.5 * (float(np.median(pos)) + float(np.median(neg)))
        predictions.extend(
            (row["observation_loss"], row[key] >= threshold)
            for row in rows if row["track"] == heldout
        )
    tp = sum(actual and predicted for actual, predicted in predictions)
    fn = sum(actual and not predicted for actual, predicted in predictions)
    tn = sum(not actual and not predicted for actual, predicted in predictions)
    fp_count = sum(not actual and predicted for actual, predicted in predictions)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp_count)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp_count},
    }


def exact_block_permutation_p(rows, key):
    """Exchange complete label sequences only among tracks of equal event count."""
    grouped = defaultdict(list)
    for track in sorted({row["track"] for row in rows}):
        track_rows = sorted((row for row in rows if row["track"] == track), key=lambda row: row["onset_sec"])
        grouped[len(track_rows)].append(track_rows)
    observed = fast_auc(
        [row[key] for row in rows if row["observation_loss"]],
        [row[key] for row in rows if not row["observation_loss"]],
    )
    null = []
    permutations = [list(itertools.permutations(range(len(blocks)))) for blocks in grouped.values()]
    for choices in itertools.product(*permutations):
        scores, labels = [], []
        for blocks, permutation in zip(grouped.values(), choices):
            for score_block, label_index in zip(blocks, permutation):
                label_block = blocks[label_index]
                scores.extend(row[key] for row in score_block)
                labels.extend(row["observation_loss"] for row in label_block)
        null.append(fast_auc([s for s, y in zip(scores, labels) if y], [s for s, y in zip(scores, labels) if not y]))
    return observed, float(sum(value >= observed for value in null) / len(null)), len(null)


def main():
    from p3_kraisler_signal_run import load_case

    with (ROOT / "results/p3_drift_cause_decomposition/events.csv").open(encoding="utf8") as handle:
        causes = {(row["track"], float(row["onset_sec"])): row["dominant_cause"] for row in csv.DictReader(handle)}
    timing = json.loads((ROOT / "results/p3_fhv_help_volatility_development/development_rows.json").read_text())["KRAISLER"]
    landmarks = {}
    for row in timing:
        if row["event"]:
            key = (row["track"], float(row["onset_sec"]))
            if key in landmarks:
                raise RuntimeError(f"duplicate event landmark: {key}")
            landmarks[key] = float(row["time_sec"])

    manifest = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text())
    cases = [row for row in manifest["rows"] if row["eligible"]]
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text())
    hop = float(config["hop_sec"])
    weights = tuple(float(value) for value in config["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        observations, reference, score_hop, _ = load_case(case, hop)
        _, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        track = case["track"]
        for key, time_sec in sorted(landmarks.items()):
            if key[0] != track:
                continue
            features = event_features(trace, int(round(time_sec / hop)), score_hop, weights, hop)
            rows.append({
                "track": track,
                "onset_sec": key[1],
                "landmark_sec": time_sec,
                "dominant_cause": causes[key],
                "observation_loss": causes[key] == "observation_loss",
                **features,
            })
        print(f"[{number}/{len(cases)}] KRAISLER-{track}", flush=True)

    features = ("help_deficit", "posterior_entropy", "negative_pmax", "negative_action_tv")
    metrics = {}
    for feature in features:
        auc = fast_auc(
            [row[feature] for row in rows if row["observation_loss"]],
            [row[feature] for row in rows if not row["observation_loss"]],
        )
        metrics[feature] = {"auc": auc, "loto_midpoint": loto_midpoint(rows, feature)}
    primary_auc, primary_p, exact_permutations = exact_block_permutation_p(rows, "help_deficit")
    metrics["help_deficit"]["exact_equal_count_track_block_permutation_p"] = primary_p
    metrics["help_deficit"]["exact_permutations"] = exact_permutations
    gates = {
        "all_events_recovered": len(rows) == 21,
        "all_positive_events_recovered": sum(row["observation_loss"] for row in rows) == 8,
        "primary_auc_ge_0_70": primary_auc >= 0.70,
        "primary_permutation_p_le_0_05": primary_p <= 0.05,
        "primary_loto_balanced_accuracy_ge_0_65": metrics["help_deficit"]["loto_midpoint"]["balanced_accuracy"] >= 0.65,
    }
    # Direction reversal was not an allowed replacement for the frozen primary
    # selector.  Preserve it explicitly as a post-failure diagnostic so a new
    # hypothesis can be named and frozen without relabeling this run.
    for row in rows:
        row["action_tv"] = -row["negative_action_tv"]
    shock_auc, shock_p, shock_permutations = exact_block_permutation_p(rows, "action_tv")
    shock_loto = loto_midpoint(rows, "action_tv")
    result = {
        "scope": "opened KRAISLER cause-selector development; not independent validation",
        "primary_selector": "negative exact 12-second help-envelope lower bound",
        "counts": {"events": len(rows), "observation_loss": sum(row["observation_loss"] for row in rows), "tracks": len({row["track"] for row in rows})},
        "metrics": metrics,
        "gates": {key: bool(value) for key, value in gates.items()},
        "selector_qualified_for_independent_freeze": bool(all(gates.values())),
        "post_failure_diagnostic": {
            "name": "observation shock (positive two-second action TV)",
            "status": "direction discovered after the frozen selector failed; requires separately frozen validation",
            "auc": shock_auc,
            "exact_equal_count_track_block_permutation_p": shock_p,
            "exact_permutations": shock_permutations,
            "loto_midpoint": shock_loto,
        },
        "original_natural_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
