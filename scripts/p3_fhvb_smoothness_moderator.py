#!/usr/bin/env python3
"""Develop an outcome-free smoothness moderator for P3-FHVB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_fhb_correction_burst_transfer as stats

OUT = ROOT / "results/p3_fhvb_smoothness_moderator"
SMOOTH_THRESHOLD = 0.03
SEED = 20260921
PERMUTATIONS = 20_000


def sources():
    development = json.loads(
        (ROOT / "results/p3_fhv_help_volatility_development/development_rows.json").read_text()
    )
    return {
        "Batik-development": development["Batik"],
        "Batik-replication": json.loads(
            (ROOT / "results/p3_fhvb_batik_replication/temporal_rows.json").read_text()
        ),
        "Batik-final": json.loads(
            (ROOT / "results/p3_fhvb6_final_validation/temporal_rows.json").read_text()
        ),
        "KRAISLER-development": development["KRAISLER"],
        "SKY-Piano-transfer": json.loads(
            (ROOT / "results/p3_sky_piano_temporal_transfer/temporal_rows.json").read_text()
        ),
    }


def track_records(source_rows):
    records = []
    for source, rows in source_rows.items():
        corpus = "Batik" if source.startswith("Batik") else "SKY-Piano" if source.startswith("SKY") else "KRAISLER"
        for track in sorted({row["track"] for row in rows}):
            group = sorted((row for row in rows if row["track"] == track), key=lambda row: row["time_sec"])
            signal = np.asarray([row["deficit"] for row in group], dtype=float)
            events = [row["deficit"] for row in group if row["event"]]
            nonevents = [row["deficit"] for row in group if not row["event"]]
            roughness = float(np.median(np.abs(np.diff(signal)))) if len(signal) > 1 else None
            auc = stats.fast_auc(events, nonevents)
            records.append(
                {
                    "source": source,
                    "corpus": corpus,
                    "track": track,
                    "cases": len(group),
                    "events": len(events),
                    "background_roughness": roughness,
                    "smooth_regime": bool(roughness is not None and roughness <= SMOOTH_THRESHOLD),
                    "auc": float(auc) if np.isfinite(auc) else None,
                }
            )
    return records


def permutation_p(x, y, observed, rng, alternative="less"):
    null = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        null[index] = spearmanr(x, rng.permutation(y)).statistic
    if alternative == "less":
        return float((1 + np.sum(null <= observed)) / (1 + len(null)))
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))


def group_summary(records, smooth):
    selected = [row for row in records if row["smooth_regime"] is smooth and row["auc"] is not None]
    return {
        "event_tracks": len(selected),
        "corpora": {corpus: sum(row["corpus"] == corpus for row in selected) for corpus in ("Batik", "KRAISLER", "SKY-Piano")},
        "mean_track_auc": float(np.mean([row["auc"] for row in selected])) if selected else None,
        "median_track_auc": float(np.median([row["auc"] for row in selected])) if selected else None,
        "track_auc_above_0_5_fraction": float(np.mean([row["auc"] > 0.5 for row in selected])) if selected else None,
    }


def main():
    records = track_records(sources())
    event_tracks = [row for row in records if row["auc"] is not None]
    roughness = np.asarray([row["background_roughness"] for row in event_tracks])
    auc = np.asarray([row["auc"] for row in event_tracks])
    rho = float(spearmanr(roughness, auc).statistic)
    rng = np.random.default_rng(SEED)
    p_value = permutation_p(roughness, auc, rho, rng)
    by_corpus = {}
    for corpus in ("Batik", "KRAISLER", "SKY-Piano"):
        group = [row for row in event_tracks if row["corpus"] == corpus]
        if len(group) >= 3:
            result = spearmanr(
                [row["background_roughness"] for row in group], [row["auc"] for row in group]
            )
            by_corpus[corpus] = {"event_tracks": len(group), "rho": float(result.statistic), "asymptotic_p": float(result.pvalue)}
    smooth = group_summary(records, True)
    rough = group_summary(records, False)
    ready = bool(
        rho <= -0.35
        and p_value <= 0.05
        and smooth["event_tracks"] >= 10
        and smooth["mean_track_auc"] >= 0.70
        and rough["mean_track_auc"] < 0.65
    )
    result = {
        "scope": "post-outcome moderator development; requires fresh validation",
        "moderator": "median absolute successive change of V(t)=abs(B_lower(t)-B_lower(t-0.5s))",
        "outcome_free_at_application": True,
        "smooth_threshold": SMOOTH_THRESHOLD,
        "event_tracks": len(event_tracks),
        "association": {"spearman_rho": rho, "track_permutation_p": p_value, "by_corpus": by_corpus},
        "smooth_regime": smooth,
        "rough_regime": rough,
        "development_gates": {
            "rho_le_minus_0_35": rho <= -0.35,
            "permutation_p_le_0_05": p_value <= 0.05,
            "smooth_event_tracks_ge_10": smooth["event_tracks"] >= 10,
            "smooth_mean_auc_ge_0_70": smooth["mean_track_auc"] >= 0.70,
            "rough_mean_auc_lt_0_65": rough["mean_track_auc"] < 0.65,
        },
        "retrospective_development_gates_passed": ready,
        "ready_for_fresh_validation": False,
        "fresh_validation_blocker": "prospective first-20-percent calibration must retain the association",
        "interpretation": "detectability moderator for P3-FHVB, not yet a causal moderator of natural drift",
        "original_directional_natural_p3_confirmed": False,
        "tracks": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "tracks"}, indent=2))


if __name__ == "__main__":
    main()
