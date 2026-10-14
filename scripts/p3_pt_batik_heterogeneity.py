#!/usr/bin/env python3
"""Diagnostic third-corpus audit of the frozen P3-PT endpoint."""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_i_natural_equivalence as natural
import p3_p_healthy_erasure as erasure
import p3_pt_kraisler_transfer as transfer
from p3_batik_external_run import load_gt
from p3_short_signal import sustained_start
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.temporal import healthy_before
import partitura as pt

MANIFEST = ROOT / "results/p3_batik_external/manifest.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_pt_batik_heterogeneity"


def load_case(case, hop):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(ROOT / case["score"]))
        notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_duration = float((notes["onset_beat"] + notes["duration_beat"]).max()) * .5
    performance_notes = load_notes(ROOT / case["performance"])
    performance_duration = max(note[1] for note in performance_notes)
    score_hop = hop * score_duration / performance_duration
    score_notes = [(float(row["onset_beat"]) * .5,
                    float(row["onset_beat"] + row["duration_beat"]) * .5,
                    int(row["pitch"])) for row in notes]
    u = notes_to_chroma_stream(performance_notes, 90., hop)
    v = notes_to_chroma_stream(score_notes, score_duration, score_hop)
    beats, perf_times = load_gt(ROOT / case["match"])
    truth = np.interp(np.arange(len(u)) * hop, perf_times, beats * .5) / score_hop
    return u, v, score_hop, truth


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    cases = [row for row in manifest["rows"] if row["eligible"]]
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8"))
    hop = float(cfg["hop_sec"])
    weights = tuple(float(value) for value in cfg["bayes"]["transition"])
    rows = []
    for number, case in enumerate(cases, 1):
        u, v, score_hop, truth = load_case(case, hop)
        cost = dose.cosine_cost(u, v)
        _, trace = dose.bayes_trace(cost, weights)
        full_error = np.abs(trace.argmax(axis=1) - truth) * score_hop
        unit = f"{case['sonata']}-{case['movement']}"
        for time_sec in range(15, 86, 5):
            end = int(round(time_sec / hop)); stop = end + 24
            if stop >= len(cost) or not healthy_before(full_error, end - 1, 20, .5):
                continue
            intact = natural.replay(trace[end - 1], cost[end:stop], weights)
            erased_cost = np.broadcast_to(cost[end:stop].mean(axis=1, keepdims=True), cost[end:stop].shape)
            erased = natural.replay(trace[end - 1], erased_cost, weights)
            local_truth = truth[end:stop]
            intact_risk = erasure.expected_risk(intact, local_truth, score_hop)
            erased_risk = erasure.expected_risk(erased, local_truth, score_hop)
            intact_error = np.abs(intact.argmax(axis=1) - local_truth) * score_hop
            erased_error = np.abs(erased.argmax(axis=1) - local_truth) * score_hop
            rows.append({"unit": unit, "sonata": case["sonata"], "time_sec": time_sec,
                         "risk_effect": float(np.mean(erased_risk[15:20] - intact_risk[15:20])),
                         "intact_event": int(sustained_start(intact_error, 0, 20) is not None),
                         "erased_event": int(sustained_start(erased_error, 0, 20) is not None)})
        print(f"[{number}/{len(cases)}] {unit}")
    effects = []
    for unit in sorted({row["unit"] for row in rows}):
        group = [row for row in rows if row["unit"] == unit]
        effects.append({"unit": unit,
                        "risk_effect": float(np.mean([row["risk_effect"] for row in group])),
                        "incidence_effect": float(np.mean([row["erased_event"] - row["intact_event"]
                                                           for row in group]))})
    rng = np.random.default_rng(20260921)
    risk = np.asarray([row["risk_effect"] for row in effects])
    incidence = np.asarray([row["incidence_effect"] for row in effects])
    result = {"scope": "post-transfer heterogeneity diagnosis; no new confirmation gate",
              "counts": {"movements": len(effects), "interventions": len(rows)},
              "metrics": {"mean_movement_risk_effect": float(risk.mean()),
                          "risk_bootstrap_95": transfer.ci(risk, rng),
                          "positive_risk_share": float(np.mean(risk > 0)),
                          "mean_movement_incidence_effect": float(incidence.mean()),
                          "incidence_bootstrap_95": transfer.ci(incidence, rng),
                          "intact_event_rate": float(np.mean([row["intact_event"] for row in rows])),
                          "erased_event_rate": float(np.mean([row["erased_event"] for row in rows]))}}
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("interventions.csv", rows), ("movement_effects.csv", effects)):
        with (OUT / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
