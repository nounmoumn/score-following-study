#!/usr/bin/env python3
"""Prospective audit of the retrospective P3-FHVB smoothness candidate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_fhb_correction_burst_transfer as stats
import p3_fhvb_smoothness_moderator as source

OUT = ROOT / "results/p3_fhvb_smoothness_prospective"
CALIBRATION_FRACTION = 0.20
NORMALIZATION_WINDOW_SEC = 12.0
STEP_SEC = 0.5
SEED = 20260921
PERMUTATIONS = 20_000


def prospective_records(sources):
    records = []
    for name, rows in sources.items():
        corpus = "Batik" if name.startswith("Batik") else "SKY-Piano" if name.startswith("SKY") else "KRAISLER"
        for track in sorted({row["track"] for row in rows}):
            group = sorted((row for row in rows if row["track"] == track), key=lambda row: row["time_sec"])
            cut = max(3, int(np.ceil(CALIBRATION_FRACTION * len(group))))
            calibration = np.asarray([row["deficit"] for row in group[:cut]])
            evaluation = group[cut:]
            events = [row["deficit"] for row in evaluation if row["event"]]
            nonevents = [row["deficit"] for row in evaluation if not row["event"]]
            auc = stats.fast_auc(events, nonevents)
            records.append(
                {
                    "source": name,
                    "corpus": corpus,
                    "track": track,
                    "calibration_cases": cut,
                    "evaluation_cases": len(evaluation),
                    "evaluation_events": len(events),
                    "calibration_roughness": float(np.median(np.abs(np.diff(calibration)))),
                    "evaluation_auc": float(auc) if np.isfinite(auc) else None,
                }
            )
    return records


def normalized_rows(sources):
    output = []
    history_count = int(round(NORMALIZATION_WINDOW_SEC / STEP_SEC))
    for name, rows in sources.items():
        corpus = "Batik" if name.startswith("Batik") else "SKY-Piano" if name.startswith("SKY") else "KRAISLER"
        for track in sorted({row["track"] for row in rows}):
            group = sorted((row for row in rows if row["track"] == track), key=lambda row: row["time_sec"])
            by_time = {round(row["time_sec"], 6): row for row in group}
            for row in group:
                times = [round(row["time_sec"] - STEP_SEC * k, 6) for k in range(history_count, 0, -1)]
                history = [by_time.get(time) for time in times]
                if any(item is None for item in history):
                    continue
                values = np.asarray([item["deficit"] for item in history])
                denominator = float(np.median(np.abs(np.diff(values))))
                output.append(
                    {
                        "corpus": corpus,
                        "track": f"{name}:{track}",
                        "normalized_signal": float(row["deficit"] / (denominator + 1e-6)),
                        "event": row["event"],
                        "onset_sec": row["onset_sec"],
                    }
                )
    return output


def main():
    sources = source.sources()
    records = prospective_records(sources)
    defined = [row for row in records if row["evaluation_auc"] is not None]
    roughness = np.asarray([row["calibration_roughness"] for row in defined])
    auc = np.asarray([row["evaluation_auc"] for row in defined])
    rho = float(spearmanr(roughness, auc).statistic)
    rng = np.random.default_rng(SEED)
    null = np.asarray([spearmanr(roughness, rng.permutation(auc)).statistic for _ in range(PERMUTATIONS)])
    p_value = float((1 + np.sum(null <= rho)) / (1 + len(null)))

    normalized = normalized_rows(sources)
    by_corpus = {}
    for corpus in ("Batik", "KRAISLER", "SKY-Piano"):
        group = [row for row in normalized if row["corpus"] == corpus]
        events = [row["normalized_signal"] for row in group if row["event"]]
        nonevents = [row["normalized_signal"] for row in group if not row["event"]]
        by_corpus[corpus] = {
            "cases": len(group),
            "events": len(events),
            "event_tracks": len({row["track"] for row in group if row["event"]}),
            "auc": stats.fast_auc(events, nonevents) if events and nonevents else None,
        }
    result = {
        "scope": "post-outcome prospective falsification audit",
        "calibration_fraction": CALIBRATION_FRACTION,
        "defined_event_tracks": len(defined),
        "prospective_association": {"spearman_rho": rho, "permutation_p": p_value},
        "prospective_moderator_retained": bool(rho <= -0.35 and p_value <= 0.05),
        "past_only_normalization": {
            "window_sec": NORMALIZATION_WINDOW_SEC,
            "corpora": by_corpus,
            "cross_corpus_gate": bool(
                all(value["auc"] is not None and value["auc"] >= 0.60 for value in by_corpus.values())
            ),
        },
        "ready_for_fresh_validation": False,
        "conclusion": "retrospective whole-track smoothness does not survive prospective calibration",
        "original_directional_natural_p3_confirmed": False,
        "tracks": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "tracks"}, indent=2))


if __name__ == "__main__":
    main()
