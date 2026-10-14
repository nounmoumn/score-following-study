"""Extract the common annotation schema from Guitar Pro 3/4/5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import guitarpro


QUARTER_TIME = 960


def techniques(note: object) -> list[str]:
    effect = note.effect
    result: list[str] = []
    if getattr(effect, "palmMute", False):
        result.append("PalmMute")
    if getattr(effect, "hammer", False):
        result.append("Hammer")
    if getattr(effect, "bend", None) is not None:
        result.append("Bend")
    for slide in getattr(effect, "slides", []) or []:
        result.append(f"Slide:{slide}")
    if getattr(effect, "letRing", False):
        result.append("LetRing")
    if getattr(effect, "vibrato", False):
        result.append("Vibrato")
    note_type = str(getattr(note, "type", "normal"))
    if note_type.lower() != "normal":
        result.append(f"NoteType:{note_type}")
    return result


def extract(path: Path) -> dict[str, Any]:
    song = guitarpro.parse(str(path))
    track_info: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    starts = [
        beat.start
        for track in song.tracks
        for measure in track.measures
        for voice in measure.voices
        for beat in voice.beats
    ]
    score_start = min(starts, default=QUARTER_TIME)

    unsupported_structure: set[str] = set()
    time_signatures: list[dict[str, int | float]] = []
    seen_time: tuple[int, int] | None = None
    measure_starts: dict[int, float] = {}
    cursor = 0.0
    if song.tracks:
        for measure in song.tracks[0].measures:
            header = measure.header
            measure_starts[header.number] = cursor
            current_time = (header.timeSignature.numerator, header.timeSignature.denominator.value)
            if current_time != seen_time:
                time_signatures.append(
                    {
                        "measure": header.number,
                        "onset_beats": cursor,
                        "numerator": current_time[0],
                        "denominator": current_time[1],
                    }
                )
                seen_time = current_time
            cursor += current_time[0] * 4 / current_time[1]
            if header.isRepeatOpen or header.repeatClose > 0 or header.repeatAlternative > 0:
                unsupported_structure.add("playback_structure:repeat")

    for track_index, track in enumerate(song.tracks):
        strings_by_number = {string.number: string.value for string in track.strings}
        tuning = [strings_by_number[number] for number in sorted(strings_by_number, reverse=True)]
        track_info.append(
            {
                "index": track_index,
                "id": str(track.number),
                "name": track.name,
                "tuning_midi_low_to_high": tuning,
                "capo_fret": track.offset,
            }
        )
        string_count = len(track.strings)
        for measure in track.measures:
            for voice_index, voice in enumerate(measure.voices):
                for beat in voice.beats:
                    change = getattr(beat.effect, "mixTableChange", None)
                    if change is not None and getattr(change, "tempo", None) is not None:
                        unsupported_structure.add("tempo_automation_after_start")
                    onset = (beat.start - score_start) / QUARTER_TIME
                    duration = beat.duration.time / QUARTER_TIME
                    for note in beat.notes:
                        open_pitch = strings_by_number[note.string]
                        expected = open_pitch + note.value
                        events.append(
                            {
                                "track": track_index,
                                "track_name": track.name,
                                "measure": measure.header.number,
                                "voice": voice_index,
                                "beat_id": str(beat.start),
                                "note_id": f"{track.number}:{measure.header.number}:{voice_index}:{beat.start}:{note.string}",
                                "onset_beats": float(onset),
                                "duration_beats": float(duration),
                                "gpif_string_index": string_count - note.string,
                                "tab_string": note.string,
                                "fret": note.value,
                                "pitch": note.realValue,
                                "expected_pitch": expected,
                                "pitch_valid": note.realValue == expected,
                                "velocity": note.velocity,
                                "dynamic": None,
                                "techniques": techniques(note),
                            }
                        )

    invalid = [event["note_id"] for event in events if not event["pitch_valid"]]
    return {
        "source": str(path.resolve()),
        "format": "legacy-gp",
        "gp_version": str(getattr(song, "version", path.suffix.lower())),
        "title": song.title or path.stem,
        "tempo_bpm_initial": float(song.tempo),
        "tempo_automations": [
            {"bar_index": 0, "position": 0.0, "bpm": float(song.tempo), "linear": False}
        ],
        "time_signatures": time_signatures,
        "unsupported_structure": sorted(unsupported_structure),
        "tracks": track_info,
        "validation": {
            "note_count": len(events),
            "pitch_mismatch_count": len(invalid),
            "pitch_mismatch_note_ids": invalid,
        },
        "notes": events,
    }
