"""Convert GPIF note annotations to a Stratus FREE force-string MIDI file."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mido


TICKS_PER_BEAT = 480
MAX_SAMPLED_FRET = 12
ARTICULATION_KEYSWITCH = {
    "sustain": 24,  # C0
    "mute": 25,     # C#0
    "choke": 26,    # D0
}


def choose_fingering(
    pitch: int,
    original_index: int,
    tuning: list[int],
    max_fret: int,
) -> tuple[int, int]:
    """Choose a playable string/fret, preferring the original and nearby strings."""
    candidates = [
        (index, pitch - open_pitch)
        for index, open_pitch in enumerate(tuning)
        if 0 <= pitch - open_pitch <= max_fret
    ]
    if not candidates:
        raise ValueError(
            f"MIDI pitch {pitch} cannot be played within frets 0-{max_fret} "
            f"for tuning {tuning}"
        )
    return min(
        candidates,
        key=lambda item: (
            item[0] != original_index,
            abs(item[0] - original_index),
            item[1],
        ),
    )


def force_string_note(tab_string: int, string_count: int) -> int:
    """Map tab string 6..1 to the default Stratus C-2..F-2 keyswitches."""
    if string_count != 6:
        raise ValueError("Stratus FREE force-string mapping requires a 6-string guitar")
    if not 1 <= tab_string <= 6:
        raise ValueError(f"Invalid tab string: {tab_string}")
    return 6 - tab_string


def articulation_for(note: dict[str, Any]) -> str:
    techniques = set(note.get("techniques", []))
    if techniques & {"Muted", "Dead", "NoteType:dead"}:
        return "choke"
    if techniques & {"PalmMuted", "PalmMute"}:
        return "mute"
    return "sustain"


def merge_tied_notes(notes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge GPIF tie destinations into their preceding sounding note."""
    result: list[dict[str, Any]] = []
    origins: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    merged = 0
    for source in sorted(
        notes,
        key=lambda note: (
            float(note["onset_beats"]), int(note["track"]), int(note["voice"]), str(note["note_id"])
        ),
    ):
        note = copy.deepcopy(source)
        key = (
            int(note["track"]),
            int(note["voice"]),
            int(note["tab_string"]),
            int(note["pitch"]),
        )
        techniques = set(note.get("techniques", []))
        if "TieDestination" in techniques and key in origins:
            origin = origins[key]
            end = float(note["onset_beats"]) + float(note["duration_beats"])
            origin["duration_beats"] = end - float(origin["onset_beats"])
            merged += 1
            if "TieOrigin" not in techniques:
                origins.pop(key, None)
            continue
        result.append(note)
        if "TieOrigin" in techniques:
            origins[key] = note
    return result, merged


def build_midi(data: dict[str, Any], max_fret: int) -> tuple[mido.MidiFile, dict[str, Any]]:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()
    midi.tracks.extend([meta_track, note_track])

    title = data.get("title") or "GPIF render"
    midi_title = title.encode("latin-1", errors="replace").decode("latin-1")
    meta_track.append(mido.MetaMessage("track_name", name=midi_title, time=0))
    tempo = float(data.get("tempo_bpm_initial") or 120.0)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    note_track.append(mido.MetaMessage("track_name", name="Shreddage 3 Stratus FREE", time=0))

    tracks = {int(track["index"]): track for track in data["tracks"]}
    events: list[tuple[int, int, int, mido.Message]] = []
    remapped: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    render_notes, merged_ties = merge_tied_notes(data["notes"])
    articulation_counts: Counter[str] = Counter()
    unsupported_techniques: Counter[str] = Counter()
    mapped_names = {
        "PalmMuted", "PalmMute", "Muted", "Dead", "NoteType:dead",
        "TieOrigin", "TieDestination",
    }

    for sequence, note in enumerate(render_notes):
        track = tracks[int(note["track"])]
        tuning = [int(value) for value in track["tuning_midi_low_to_high"]]
        pitch = int(note["pitch"])
        original_index = int(note["gpif_string_index"])
        try:
            render_index, render_fret = choose_fingering(
                pitch, original_index, tuning, max_fret
            )
        except ValueError as exc:
            skipped.append(
                {
                    "note_id": note["note_id"],
                    "pitch": pitch,
                    "original_tab_string": note["tab_string"],
                    "original_fret": note["fret"],
                    "reason": str(exc),
                }
            )
            continue

        render_tab_string = len(tuning) - render_index
        tick = round(float(note["onset_beats"]) * TICKS_PER_BEAT)
        end_tick = round(
            (float(note["onset_beats"]) + float(note["duration_beats"]))
            * TICKS_PER_BEAT
        )
        velocity = max(1, min(127, int(note["velocity"])))
        keyswitch = force_string_note(render_tab_string, len(tuning))
        articulation = articulation_for(note)
        articulation_note = ARTICULATION_KEYSWITCH[articulation]
        articulation_counts[articulation] += 1
        for technique in note.get("techniques", []):
            if technique not in mapped_names:
                unsupported_techniques[technique] += 1

        # Same-tick ordering is intentional: latch the string, release the
        # keyswitch, then play the musical note. This also handles chords by
        # selecting a string immediately before each chord tone.
        priority = sequence * 5
        events.append((tick, 1, priority, mido.Message("note_on", note=articulation_note, velocity=100)))
        events.append((tick, 1, priority + 1, mido.Message("note_off", note=articulation_note, velocity=0)))
        events.append((tick, 1, priority + 2, mido.Message("note_on", note=keyswitch, velocity=100)))
        events.append((tick, 1, priority + 3, mido.Message("note_off", note=keyswitch, velocity=0)))
        events.append((tick, 1, priority + 4, mido.Message("note_on", note=pitch, velocity=velocity)))
        # Note-offs always precede new attacks at the same tick.
        events.append((end_tick, 0, sequence, mido.Message("note_off", note=pitch, velocity=0)))

        if render_index != original_index:
            remapped.append(
                {
                    "note_id": note["note_id"],
                    "pitch": pitch,
                    "original_tab_string": note["tab_string"],
                    "original_fret": note["fret"],
                    "render_tab_string": render_tab_string,
                    "render_fret": render_fret,
                }
            )

    events.sort(key=lambda item: (item[0], item[1]))
    previous_tick = 0
    for tick, _, _, message in events:
        message.time = tick - previous_tick
        note_track.append(message)
        previous_tick = tick
    note_track.append(mido.MetaMessage("end_of_track", time=0))

    report = {
        "source": data.get("source"),
        "max_sampled_fret": max_fret,
        "input_note_count": len(data["notes"]),
        "rendered_note_count": len(render_notes) - len(skipped),
        "tie_destination_notes_merged": merged_ties,
        "remapped_note_count": len(remapped),
        "skipped_note_count": len(skipped),
        "remapped_notes": remapped,
        "skipped_notes": skipped,
        "articulation_counts": dict(articulation_counts),
        "unsupported_techniques": dict(unsupported_techniques),
    }
    return midi, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path, help="*.notes.json from gpif_to_json.py")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--max-fret", type=int, default=MAX_SAMPLED_FRET)
    args = parser.parse_args()

    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    output = args.output or args.annotation.with_name(
        args.annotation.name.removesuffix(".notes.json") + ".shreddage.mid"
    )
    midi, report = build_midi(data, args.max_fret)
    midi.save(output)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote MIDI: {output.resolve()}")
    print(f"Wrote report: {report_path.resolve()}")
    print(f"Rendered notes: {report['rendered_note_count']}")
    print(f"Remapped notes: {report['remapped_note_count']}")
    print(f"Skipped notes: {report['skipped_note_count']}")


if __name__ == "__main__":
    main()
