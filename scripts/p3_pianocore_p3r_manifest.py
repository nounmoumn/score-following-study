#!/usr/bin/env python3
"""Outcome-blind PianoCoRe manifests for P3-R development and preservation."""

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p3_external_metadata/PianoCoRe_metadata.csv"
OUT = ROOT / "results/p3_pianocore_p3r_manifest.json"
USED_SURNAMES = {
    "bach", "balakirev", "beethoven", "brahms", "chopin", "debussy",
    "glinka", "haydn", "liszt", "mozart", "prokofiev", "rachmaninoff",
    "ravel", "schubert", "schumann", "scriabin",
}
SOURCES = ("GiantMIDI-Piano", "ATEPP", "PERiScoPe", "Aria-MIDI")
VALIDATION_SIZE = 17
DEVELOPMENT_SIZE = 30


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def surname(composer):
    return re.sub(r"[^a-z]", "", composer.split(",", 1)[0].lower())


def eligible(row):
    duration = float(row.get("performance_duration") or 0)
    return (
        truth(row.get("tier_a_star"))
        and row.get("performance_dataset") != "ASAP"
        and row.get("score_dataset") != "ASAP"
        and 90 <= duration <= 600
        and bool(row.get("performance_midi_path"))
        and bool(row.get("score_midi_path"))
        and bool(row.get("raw_alignment_path"))
        and not truth(row.get("is_duplicate"))
        and float(row.get("raw_precision") or 0) >= 0.90
        and float(row.get("raw_recall") or 0) >= 0.90
        and surname(row.get("composer", "")) not in USED_SURNAMES
    )


def public(row):
    fields = (
        "id", "composer", "composition", "movement", "split", "performance_dataset",
        "score_dataset", "score_id", "performance_midi_path", "score_midi_path",
        "raw_alignment_path", "performance_duration", "score_duration",
        "raw_precision", "raw_recall", "is_transcription",
    )
    return {field: row.get(field, "") for field in fields} | {
        "dataset_viewer_offset": int(row["_split_offset"]),
        "work_key": [row.get("composer", ""), row.get("composition", ""), row.get("movement", "")],
    }


def select(best, split, target, forbidden_composers=frozenset()):
    chosen, used = [], set(forbidden_composers)
    # Guarantee source coverage first; fill remaining slots by alignment quality.
    for source in SOURCES:
        pool = [row for (s, src, _), row in best.items() if s == split and src == source]
        pool.sort(key=lambda row: (-row["_quality"], row["composer"], row["composition"], row["movement"], row["id"]))
        for row in pool:
            if row["composer"] in used:
                continue
            chosen.append(row); used.add(row["composer"])
            break
        else:
            raise RuntimeError(f"no unique composer for {split} {source}")
    pool = [row for (s, _, _), row in best.items() if s == split]
    pool.sort(key=lambda row: (-row["_quality"], row["composer"], row["composition"], row["movement"], row["id"]))
    for row in pool:
        if row["composer"] not in used:
            chosen.append(row); used.add(row["composer"])
            if len(chosen) == target:
                break
    if len(chosen) != target:
        raise RuntimeError(f"could not fill {split}: {len(chosen)}/{target}")
    return chosen


def main():
    best, offsets = {}, {"train": 0, "test": 0}
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            row["_split_offset"] = offsets.get(split, 0)
            offsets[split] = offsets.get(split, 0) + 1
            if not eligible(row):
                continue
            key = (split, row["performance_dataset"], row["composer"])
            row["_quality"] = min(float(row["raw_precision"]), float(row["raw_recall"]))
            previous = best.get(key)
            if previous is None or (row["_quality"], row["id"]) > (previous["_quality"], previous["id"]):
                best[key] = row

    validation = select(best, "test", VALIDATION_SIZE)
    validation_composers = {row["composer"] for row in validation}
    development = select(best, "train", DEVELOPMENT_SIZE, validation_composers)
    if validation_composers & {row["composer"] for row in development}:
        raise AssertionError("composer leakage")
    result = {
        "scope": "Outcome-blind PianoCoRe P3-R file manifest; no tracker outcomes computed",
        "source": "https://huggingface.co/datasets/SyMuPe/PianoCoRe",
        "metadata_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "criteria": {
            "tier_a_star": True, "non_asap_score_and_performance": True,
            "duration_sec": [90, 600], "raw_precision_min": 0.90,
            "raw_recall_min": 0.90, "is_duplicate": False,
            "excluded_asap_development_surnames": sorted(USED_SURNAMES),
            "one_per_composer": True,
        },
        "selection": {"development_size": DEVELOPMENT_SIZE, "validation_size": VALIDATION_SIZE,
                      "minimum_one_per_source": list(SOURCES), "remaining_order": "raw alignment quality descending"},
        "development": [public(row) for row in development],
        "preserved_validation": [public(row) for row in validation],
        "counts": {
            "development": len(development), "development_composers": len({r["composer"] for r in development}),
            "validation": len(validation), "validation_composers": len(validation_composers),
            "composer_overlap": 0,
            "development_sources": dict(__import__("collections").Counter(r["performance_dataset"] for r in development)),
            "validation_sources": dict(__import__("collections").Counter(r["performance_dataset"] for r in validation)),
        },
        "caveat": "All selected performances are automatic transcriptions. Suitable for artificial P3-R development/transfer only, not natural-P3 timing ground truth.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("development", "preserved_validation")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
