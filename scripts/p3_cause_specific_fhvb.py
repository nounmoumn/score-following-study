#!/usr/bin/env python3
"""Post-outcome check of P3-FHVB by exact retrospective drift cause."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_fhb_correction_burst_transfer as stats

OUT = ROOT / "results/p3_cause_specific_fhvb"
SEED = 20260921
PERMUTATIONS = 2_000
CAUSES = ("observation_loss", "transition", "occurrence")


def evaluate(cause, rows, labels, rng):
    selected = []
    for row in rows:
        if row["event"]:
            actual = labels[(row["track"], float(row["onset_sec"]))]
            if actual != cause:
                continue
            selected.append({**row, "event": True})
        else:
            selected.append({**row, "event": False})
    events = [row["deficit"] for row in selected if row["event"]]
    nonevents = [row["deficit"] for row in selected if not row["event"]]
    auc = stats.fast_auc(events, nonevents)
    null = np.asarray([stats.fast_shift_auc(selected, rng) for _ in range(PERMUTATIONS)])
    null = null[np.isfinite(null)]
    p_value = float((1 + np.sum(null >= auc)) / (1 + len(null)))
    return {
        "cause": cause,
        "events": len(events),
        "event_tracks": len({row["track"] for row in selected if row["event"]}),
        "event_median": float(np.median(events)),
        "nonevent_median": float(np.median(nonevents)),
        "auc": auc,
        "within_track_shift_p": p_value,
    }


def main():
    with (ROOT / "results/p3_drift_cause_decomposition/events.csv").open(encoding="utf8") as handle:
        labels = {(row["track"], float(row["onset_sec"])): row["dominant_cause"] for row in csv.DictReader(handle)}
    source = json.loads(
        (ROOT / "results/p3_fhv_help_volatility_development/development_rows.json").read_text()
    )["KRAISLER"]
    rng = np.random.default_rng(SEED)
    cause_results = [evaluate(cause, source, labels, rng) for cause in CAUSES]
    event_rows = [row for row in source if row["event"]]
    observation = [row["deficit"] for row in event_rows if labels[(row["track"], float(row["onset_sec"]))] == "observation_loss"]
    other = [row["deficit"] for row in event_rows if labels[(row["track"], float(row["onset_sec"]))] != "observation_loss"]
    result = {
        "scope": "post-outcome cause-specific diagnostic; not independent confirmation",
        "signal": "unchanged P3-FHVB help volatility",
        "causes": cause_results,
        "observation_loss_vs_other_event_auc": stats.fast_auc(observation, other),
        "pattern_consistent_with_p3_dc": bool(
            cause_results[0]["auc"] >= 0.65
            and cause_results[0]["auc"] > cause_results[1]["auc"]
            and cause_results[1]["auc"] > cause_results[2]["auc"]
        ),
        "selector_validated": False,
        "original_directional_natural_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
