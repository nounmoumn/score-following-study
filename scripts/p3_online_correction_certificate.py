#!/usr/bin/env python3
"""GT-free exact correction-certificate diagnostic on opened event sets."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fci_tight_envelope_diagnostic as tight
import p3_observation_loss_selector as selector
import p3_vienna_forecast_validation as vienna
from p3_fhb_correction_burst_transfer import fast_auc

OUT = ROOT / "results/p3_online_correction_certificate"
RADIUS = 0.5211881394162992
HISTORY_SEC = 2.0
CAP = 5.0


def correction_scores(trace, end, score_hop, weights, hop):
    upper_sum = 0.0
    point_sum = 0.0
    states = np.arange(trace.shape[1], dtype=float)
    for frame in range(end - int(round(HISTORY_SEC / hop)), end):
        prior = dc.apply_prediction(trace[frame - 1], weights)
        delta = trace[frame] - prior
        center = selector.median_state(trace[frame])
        _, correction_upper = tight.help_envelope(delta, center, RADIUS, score_hop, CAP)
        upper_sum += correction_upper
        distance = np.minimum(np.abs(states - center) * score_hop, CAP)
        point_sum += -float(distance @ delta)
    return {"certificate_score": -float(upper_sum), "point_score": -float(point_sum)}


def certificate_score(trace, end, score_hop, weights, hop):
    return correction_scores(trace, end, score_hop, weights, hop)["certificate_score"]


def summarize(rows):
    positive = [row["certificate_score"] for row in rows if row["observation_loss"]]
    negative = [row["certificate_score"] for row in rows if not row["observation_loss"]]
    return {
        "events": len(rows), "observation_loss": len(positive),
        "positive_certificates": sum(row["certificate_score"] > 0 for row in rows),
        "observation_loss_positive_certificates": sum(
            row["certificate_score"] > 0 and row["observation_loss"] for row in rows
        ),
        "observation_loss_median": float(np.median(positive)),
        "other_median": float(np.median(negative)),
        "auc": fast_auc(positive, negative),
    }


def main():
    config = yaml.safe_load((ROOT / "configs/p3_temporal.yaml").read_text())
    hop = float(config["hop_sec"]); weights = tuple(float(x) for x in config["bayes"]["transition"])
    outputs = []

    from p3_kraisler_signal_run import load_case
    with (ROOT / "results/p3_observation_loss_selector/events.csv").open(encoding="utf8") as handle:
        k_events = list(csv.DictReader(handle))
    manifest = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text())
    for case in (row for row in manifest["rows"] if row["eligible"]):
        selected = [row for row in k_events if row["track"] == case["track"]]
        if not selected: continue
        observations, reference, score_hop, _ = load_case(case, hop)
        _, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        for row in selected:
            outputs.append({"corpus": "KRAISLER", "track": row["track"], "onset_sec": float(row["onset_sec"]),
                            "observation_loss": row["observation_loss"] == "True",
                            **correction_scores(trace, int(round(float(row["landmark_sec"]) / hop)), score_hop, weights, hop)})
        print(f"[KRAISLER] {case['track']}", flush=True)

    with (ROOT / "results/p3_vienna_observation_shock_validation/events.csv").open(encoding="utf8") as handle:
        v_events = list(csv.DictReader(handle))
    v_manifest = json.loads((ROOT / "results/p3_vienna_forecast_preflight/manifest.json").read_text())
    cache = {}
    for case in (row for row in v_manifest["rows"] if row["split"] == "validation"):
        selected = [row for row in v_events if row["track"] == case["stem"]]
        if not selected: continue
        observations, reference, score_hop, _, _ = vienna.load_case(case, hop, cache)
        _, trace = dose.bayes_trace(dose.cosine_cost(observations, reference), weights)
        for row in selected:
            outputs.append({"corpus": "Vienna", "track": row["track"], "onset_sec": float(row["onset_sec"]),
                            "observation_loss": row["observation_loss"] == "True",
                            **correction_scores(trace, int(round(float(row["time_sec"]) / hop)), score_hop, weights, hop)})
        print(f"[Vienna] {case['stem']}", flush=True)

    corpora = {name: summarize([row for row in outputs if row["corpus"] == name]) for name in ("KRAISLER", "Vienna")}
    pooled = summarize(outputs)
    point_diagnostic = {}
    for name in ("KRAISLER", "Vienna", "pooled"):
        rows = outputs if name == "pooled" else [row for row in outputs if row["corpus"] == name]
        positive = [row["point_score"] for row in rows if row["observation_loss"]]
        negative = [row["point_score"] for row in rows if not row["observation_loss"]]
        point_diagnostic[name] = {"auc": fast_auc(positive, negative),
                                  "observation_loss_median": float(np.median(positive)),
                                  "other_median": float(np.median(negative))}
    gates = {"each_corpus_auc_ge_0_65": all(value["auc"] >= 0.65 for value in corpora.values()),
             "pooled_auc_ge_0_70": pooled["auc"] >= 0.70}
    result = {"scope": "opened-corpus development diagnostic; not validation", "corpora": corpora,
              "pooled": pooled, "post_failure_posterior_median_diagnostic": point_diagnostic,
              "gates": gates, "qualified_for_future_independent_freeze": bool(all(gates.values())),
              "natural_p3_confirmed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "events.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0])); writer.writeheader(); writer.writerows(outputs)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
