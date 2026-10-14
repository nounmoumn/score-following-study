#!/usr/bin/env python3
"""Outcome-blind inventory and eligibility freeze for CP-WTC symbolic data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes

DATA = ROOT / "data/cp_wtc/symbolic"
OUT = ROOT / "results/p3_cp_wtc_preflight"

# ASAP25 contained both prelude and fugue for these WTC-I BWV numbers.
ASAP_BWV = {846, 848, 854, 856, 857, 858, 860, 862, 863, 864, 865, 866, 867, 868}
ASAP_PIECES = {2 * (bwv - 846) + offset for bwv in ASAP_BWV for offset in (1, 2)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_case(artist: str, midi: Path) -> dict:
    piece = int(midi.stem)
    alignment = DATA / "alignments" / artist / f"{piece:02d}.txt"
    score = DATA / "scores" / f"{piece:02d}.musicxml"
    points = np.loadtxt(alignment)
    notes = load_notes(midi)
    duration = float(max(note[1] for note in notes))
    score_clock = points[:, 0]
    perf_clock = points[:, 1]
    eligible = bool(
        score.exists()
        and len(points) >= 2
        and np.all(np.diff(score_clock) > 0)
        and np.all(np.diff(perf_clock) > 0)
        and duration >= 30.0
        and piece not in ASAP_PIECES
    )
    return {
        "artist": artist,
        "piece": f"{piece:02d}",
        "duration_sec": duration,
        "alignment_rows": int(len(points)),
        "perf_clock_monotone": bool(np.all(np.diff(perf_clock) > 0)),
        "score_clock_monotone": bool(np.all(np.diff(score_clock) > 0)),
        "score_exists": score.exists(),
        "overlaps_asap25_composition": piece in ASAP_PIECES,
        "eligible": eligible,
        "midi": str(midi.relative_to(ROOT)).replace("\\", "/"),
        "alignment": str(alignment.relative_to(ROOT)).replace("\\", "/"),
        "score": str(score.relative_to(ROOT)).replace("\\", "/"),
        "midi_sha256": digest(midi),
        "alignment_sha256": digest(alignment),
        "score_sha256": digest(score),
    }


def main() -> None:
    rows = []
    for artist_dir in sorted((DATA / "midi").iterdir()):
        if artist_dir.is_dir():
            rows.extend(inspect_case(artist_dir.name, path) for path in sorted(artist_dir.glob("*.midi")))
    eligible = [row for row in rows if row["eligible"]]
    summary = {
        "dataset": "CP-WTC symbolic subset",
        "source_commit": "9fe2bc1c1dae83fd5cc89e5db6b27e44a4e281d3",
        "known_limitations": [
            "performance MIDI is Transkun automatic transcription, not direct-capture ground truth",
            "score-to-performance alignment is automatic",
            "only the first eight bars are represented",
        ],
        "inventory": {
            "performances": len(rows),
            "artists": len({row["artist"] for row in rows}),
            "pieces": len({row["piece"] for row in rows}),
        },
        "eligibility_rule": "duration>=30s, both clocks strictly increasing, files present, no ASAP25 WTC composition overlap",
        "eligible": {
            "performances": len(eligible),
            "artists": len({row["artist"] for row in eligible}),
            "pieces": len({row["piece"] for row in eligible}),
            "piece_ids": sorted({row["piece"] for row in eligible}),
        },
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({key: summary[key] for key in ("inventory", "eligibility_rule", "eligible")}, indent=2))


if __name__ == "__main__":
    main()
