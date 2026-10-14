#!/usr/bin/env python3
"""Freeze an outcome-blind ASAP benchmark manifest for the path clock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASAP = ROOT / "asap_dataset"
OUT = ROOT / "results/p3_natural_path_clock_manifest.json"
TARGET = 24


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.replace("\\", "/").encode("utf8")).hexdigest()


def eligible_rows():
    rows = []
    for score_annotations in ASAP.rglob("midi_score_annotations.txt"):
        directory = score_annotations.parent
        relative_directory = directory.relative_to(ASAP).as_posix()
        if relative_directory == "Bach/Fugue/bwv_875":
            continue
        score_midi = directory / "midi_score.mid"
        if not score_midi.exists():
            continue
        candidates = []
        for annotations in directory.glob("*_annotations.txt"):
            if annotations.name == "midi_score_annotations.txt":
                continue
            midi = directory / (annotations.name.removesuffix("_annotations.txt") + ".mid")
            if midi.exists():
                key = text_digest(midi.relative_to(ROOT).as_posix())
                candidates.append((key, midi, annotations))
        if not candidates:
            continue
        _, performance, performance_annotations = min(candidates)
        composer = relative_directory.split("/")[0]
        rows.append({
            "composer": composer,
            "work": relative_directory,
            "work_order_digest": text_digest(relative_directory),
            "score_midi": score_midi.relative_to(ROOT).as_posix(),
            "score_annotations": score_annotations.relative_to(ROOT).as_posix(),
            "performance_midi": performance.relative_to(ROOT).as_posix(),
            "performance_annotations": performance_annotations.relative_to(ROOT).as_posix(),
        })
    return rows


def select(rows):
    by_composer = {}
    for row in rows:
        by_composer.setdefault(row["composer"], []).append(row)
    for items in by_composer.values():
        items.sort(key=lambda row: row["work_order_digest"])
    composers = sorted(by_composer, key=text_digest)
    selected = []
    round_number = 0
    while len(selected) < TARGET:
        added = False
        for composer in composers:
            if round_number < len(by_composer[composer]):
                selected.append(by_composer[composer][round_number])
                added = True
                if len(selected) == TARGET:
                    break
        if not added:
            break
        round_number += 1
    return selected


def main():
    eligible = eligible_rows()
    selected = select(eligible)
    if len(selected) != TARGET or len({row["composer"] for row in selected}) < 6:
        raise RuntimeError("Frozen benchmark availability gate failed")
    for row in selected:
        row["sha256"] = {
            key: sha256(ROOT / row[key])
            for key in ("score_midi", "score_annotations", "performance_midi", "performance_annotations")
        }
    result = {
        "scope": "Outcome-blind ASAP beat-alignment benchmark; no tracker/P3 outcomes",
        "selection_algorithm": "path-hash performance; composer round-robin work-hash; BWV875 excluded",
        "eligible_works": len(eligible),
        "selected_works": len(selected),
        "selected_composers": len({row["composer"] for row in selected}),
        "rows": selected,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
