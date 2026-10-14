#!/usr/bin/env python3
"""Post-outcome audit of P3-FHVB heterogeneity across corpora."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_fhb_correction_burst_transfer as stats

OUT = ROOT / "results/p3_fhvb_corpus_heterogeneity"
SEED = 20260921
PERMUTATIONS = 10_000


def load_rows():
    development = json.loads(
        (ROOT / "results/p3_fhv_help_volatility_development/development_rows.json").read_text()
    )
    sources = {
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
    rows = []
    for source, source_rows in sources.items():
        corpus = source.split("-", 1)[0]
        if source.startswith("SKY"):
            corpus = "SKY-Piano"
        for row in source_rows:
            rows.append({**row, "source": source, "corpus": corpus, "track_key": f"{source}:{row['track']}"})
    return rows


def pooled_auc(rows):
    return stats.fast_auc(
        [row["deficit"] for row in rows if row["event"]],
        [row["deficit"] for row in rows if not row["event"]],
    )


def corpus_summary(rows, corpus):
    selected = [row for row in rows if row["corpus"] == corpus]
    track_keys = sorted({row["track_key"] for row in selected})
    track_auc = []
    for key in track_keys:
        group = [row for row in selected if row["track_key"] == key]
        value = pooled_auc(group)
        if np.isfinite(value):
            track_auc.append(float(value))
    return {
        "tracks": len(track_keys),
        "event_tracks_with_defined_auc": len(track_auc),
        "cases": len(selected),
        "unique_onsets": len({(row["track_key"], row["onset_sec"]) for row in selected if row["event"]}),
        "pooled_auc": pooled_auc(selected),
        "median_event_track_auc": float(np.median(track_auc)) if track_auc else None,
        "event_track_auc_above_0_5_fraction": float(np.mean(np.asarray(track_auc) > 0.5)) if track_auc else None,
    }


def interaction_permutation(rows, rng):
    packages = {}
    for row in rows:
        packages.setdefault(row["track_key"], []).append(row)
    items = []
    for key, group in sorted(packages.items()):
        value = pooled_auc(group)
        if np.isfinite(value):
            items.append((key, group[0]["corpus"], float(value)))
    values = np.asarray([item[2] for item in items])
    batik = np.asarray([item[1] == "Batik" for item in items])
    batik_count = int(batik.sum())
    observed = float(values[batik].mean() - values[~batik].mean())
    null = []
    for _ in range(PERMUTATIONS):
        chosen = rng.choice(len(values), size=batik_count, replace=False)
        mask = np.zeros(len(values), dtype=bool)
        mask[chosen] = True
        null.append(float(values[mask].mean() - values[~mask].mean()))
    p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))
    return observed, p_value


def main():
    rows = load_rows()
    summaries = {corpus: corpus_summary(rows, corpus) for corpus in ("Batik", "KRAISLER", "SKY-Piano")}
    difference, p_value = interaction_permutation(rows, np.random.default_rng(SEED))
    result = {
        "scope": "post-outcome descriptive corpus heterogeneity audit; not a new confirmation test",
        "signal": "frozen 12-second-window, 0.5-second-projection P3-FHVB",
        "corpora": summaries,
        "batik_minus_non_batik_mean_event_track_auc": difference,
        "track_label_permutation_p": p_value,
        "permutations": PERMUTATIONS,
        "conclusion": "corpus moderation is material" if p_value <= 0.05 else "corpus moderation unresolved",
        "original_directional_natural_p3_confirmed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
