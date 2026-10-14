"""Batch-build annotations, Stratus MIDI, reports, and a dataset manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from gpif_to_json import extract as extract_gpif
from legacy_gp_to_json import extract as extract_legacy
from notes_to_shreddage_midi import build_midi


SUPPORTED = {".gp", ".gp3", ".gp4", ".gp5"}


def safe_id(path: Path) -> str:
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)
    readable = "_".join(part for part in readable.split("_") if part)[:80] or "score"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


def extract_score(path: Path) -> dict[str, Any]:
    if zipfile.is_zipfile(path):
        data = extract_gpif(path)
        data["format"] = "gpif"
        return data
    return extract_legacy(path)


def summarize(
    source: Path,
    item_id: str,
    data: dict[str, Any],
    render_report: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    notes = data.get("notes", [])
    techniques = Counter(t for note in notes for t in note.get("techniques", []))
    skipped = render_report["skipped_note_count"] if render_report else None
    mismatches = data.get("validation", {}).get("pitch_mismatch_count", 0)
    unsupported_structure = data.get("unsupported_structure", [])
    if error:
        status = "error"
    elif mismatches:
        status = "rejected_pitch_mismatch"
    elif unsupported_structure:
        status = "unsupported_structure"
    elif skipped:
        status = "partial_out_of_range"
    elif render_report and render_report["remapped_note_count"]:
        status = "ready_with_string_remap"
    else:
        status = "ready_exact_fingering"
    return {
        "id": item_id,
        "source": str(source.resolve()),
        "format": data.get("format"),
        "title": data.get("title"),
        "status": status,
        "note_count": len(notes),
        "track_count": len(data.get("tracks", [])),
        "max_fret": max((note["fret"] for note in notes), default=None),
        "pitch_mismatch_count": mismatches,
        "unsupported_structure": unsupported_structure,
        "remapped_note_count": render_report["remapped_note_count"] if render_report else None,
        "skipped_note_count": skipped,
        "techniques": dict(sorted(techniques.items())),
        "error": error,
    }


def process(source: Path, output_root: Path, max_fret: int) -> dict[str, Any]:
    item_id = safe_id(source)
    item_dir = output_root / "items" / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = item_dir / "notes.json"
    midi_path = item_dir / "shreddage.mid"
    report_path = item_dir / "render_report.json"
    metadata_path = item_dir / "metadata.json"
    source_copy = item_dir / ("source" + source.suffix.lower())

    try:
        data = extract_score(source)
        annotation_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if source.resolve() != source_copy.resolve():
            shutil.copy2(source, source_copy)
        midi, render_report = build_midi(data, max_fret)
        midi.save(midi_path)
        report_path.write_text(json.dumps(render_report, indent=2), encoding="utf-8")
        row = summarize(source, item_id, data, render_report)
    except Exception as exc:  # keep a complete batch report instead of stopping at one file
        row = summarize(source, item_id, {}, None, f"{type(exc).__name__}: {exc}")
    metadata_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def write_manifest(rows: list[dict[str, Any]], output_root: Path) -> None:
    (output_root / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "id", "source", "format", "title", "status", "note_count", "track_count",
        "max_fret", "pitch_mismatch_count", "unsupported_structure", "remapped_note_count", "skipped_note_count",
        "techniques", "error",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["techniques"] = json.dumps(row["techniques"], ensure_ascii=False)
            csv_row["unsupported_structure"] = json.dumps(
                row["unsupported_structure"], ensure_ascii=False
            )
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Score file or directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/guitar_dataset"))
    parser.add_argument("--max-fret", type=int, default=12)
    args = parser.parse_args()

    sources = (
        [args.input]
        if args.input.is_file()
        else sorted(
            path for path in args.input.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED
        )
    )
    if not sources:
        raise SystemExit(f"No supported Guitar Pro files found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, source in enumerate(sources, 1):
        print(f"[{index}/{len(sources)}] {source}")
        row = process(source, args.output, args.max_fret)
        rows.append(row)
        print(f"  {row['status']}: {row['note_count']} notes")
    write_manifest(rows, args.output)
    counts = Counter(row["status"] for row in rows)
    print(f"Manifest: {(args.output / 'manifest.csv').resolve()}")
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if counts.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
