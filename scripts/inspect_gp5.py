"""Inspect Guitar Pro notes without discarding string/fret information."""

from __future__ import annotations

import argparse
from pathlib import Path

import guitarpro


SUPPORTED_SUFFIXES = {".gp", ".gp3", ".gp4", ".gp5", ".gpx"}


def effect_summary(note: object) -> str:
    """Return the most useful note effects in a compact, version-safe form."""
    effect = note.effect
    bend = getattr(effect, "bend", None)
    slides = getattr(effect, "slides", []) or []
    return (
        f"palm_mute={bool(getattr(effect, 'palmMute', False))} | "
        f"hammer={bool(getattr(effect, 'hammer', False))} | "
        f"slides={','.join(str(slide) for slide in slides) or '-'} | "
        f"bend={bend is not None}"
    )


def inspect_file(path: Path, max_notes: int) -> None:
    with path.open("rb") as stream:
        if stream.read(4).startswith(b"PK"):
            raise SystemExit(
                "This is a ZIP-based Guitar Pro 7+ .gp file, which PyGuitarPro "
                "cannot read. Open it in TuxGuitar and export/save a copy as "
                "Guitar Pro 5 (.gp5), then run this command on that copy."
            )
    song = guitarpro.parse(str(path))
    print(f"File: {path}")
    print(f"Title: {song.title or '(untitled)'}")
    print(f"Tracks: {len(song.tracks)}")

    for track in song.tracks:
        print("\n" + "=" * 72)
        print(f"Track {track.number}: {track.name}")
        print("Tuning (string 1 is the highest-pitched string):")
        for string in track.strings:
            print(f"  string {string.number}: MIDI {string.value}")
        print(f"Capo: {track.offset}")

        count = 0
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        print(
                            f"measure={measure.header.number:3d} | "
                            f"beat_start={beat.start:6d} | "
                            f"string={note.string} | "
                            f"fret={note.value:2d} | "
                            f"pitch={note.realValue:3d} | "
                            f"velocity={note.velocity:3d} | "
                            f"{effect_summary(note)}"
                        )
                        count += 1
                        if count >= max_notes:
                            break
                    if count >= max_notes:
                        break
                if count >= max_notes:
                    break
            if count >= max_notes:
                break
        print(f"Shown notes: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print string, fret, pitch, timing, and effects from Guitar Pro files."
    )
    parser.add_argument("file", type=Path, help="Path to a Guitar Pro file")
    parser.add_argument(
        "--max-notes", type=int, default=30, help="Maximum notes shown per track"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.file.resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise SystemExit(f"Unsupported extension: {path.suffix}")
    if args.max_notes < 1:
        raise SystemExit("--max-notes must be at least 1")
    inspect_file(path, args.max_notes)


if __name__ == "__main__":
    main()
