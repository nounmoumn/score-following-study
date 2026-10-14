"""Extract lossless guitar-note annotations from a Guitar Pro 7+ ``.gp`` file.

The modern ``.gp`` format is a ZIP container. Its ``Content/score.gpif`` member
is XML and keeps string/fret information that ordinary MIDI cannot represent.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NOTE_BEATS = {
    "Whole": Fraction(4),
    "Half": Fraction(2),
    "Quarter": Fraction(1),
    "Eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
    "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16),
    "128th": Fraction(1, 32),
}

DYNAMIC_VELOCITY = {
    "PPP": 15,
    "PP": 31,
    "P": 47,
    "MP": 63,
    "MF": 79,
    "F": 95,
    "FF": 111,
    "FFF": 127,
}


def required_text(node: ET.Element, path: str) -> str:
    child = node.find(path)
    if child is None or child.text is None:
        raise ValueError(f"Missing required GPIF field: {path}")
    return child.text.strip()


def id_map(parent: ET.Element | None) -> dict[str, ET.Element]:
    if parent is None:
        return {}
    return {required_text_as_attr(child, "id"): child for child in parent}


def required_text_as_attr(node: ET.Element, name: str) -> str:
    value = node.get(name)
    if value is None:
        raise ValueError(f"Missing required GPIF attribute: {node.tag}@{name}")
    return value


def refs(node: ET.Element, path: str) -> list[str]:
    child = node.find(path)
    if child is None or not child.text:
        return []
    return [value for value in child.text.split() if value != "-1"]


def rhythm_beats(rhythm: ET.Element) -> Fraction:
    name = required_text(rhythm, "NoteValue")
    try:
        duration = NOTE_BEATS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported GPIF NoteValue: {name}") from exc

    dot = rhythm.find("AugmentationDot")
    dot_count = int(dot.get("count", "0")) if dot is not None else 0
    addition = duration
    for _ in range(dot_count):
        addition /= 2
        duration += addition

    tuplet = rhythm.find("PrimaryTuplet")
    if tuplet is not None:
        numerator = int(tuplet.get("num", "0"))
        denominator = int(tuplet.get("den", "0"))
        if numerator <= 0 or denominator <= 0:
            raise ValueError("Invalid GPIF PrimaryTuplet")
        duration *= Fraction(denominator, numerator)
    return duration


def property_element(note: ET.Element, name: str, value_tag: str) -> ET.Element | None:
    for prop in note.findall("./Properties/Property"):
        if prop.get("name") == name:
            return prop.find(value_tag)
    return None


def property_int(note: ET.Element, name: str, value_tag: str) -> int:
    value = property_element(note, name, value_tag)
    if value is None or value.text is None:
        raise ValueError(f"Note {note.get('id')} lacks {name}/{value_tag}")
    return int(value.text)


def note_techniques(note: ET.Element) -> list[str]:
    techniques: list[str] = []
    properties = note.find("Properties")
    if properties is not None:
        for prop in properties.findall("Property"):
            name = prop.get("name")
            if name and name not in {"ConcertPitch", "TransposedPitch", "String", "Fret", "Midi"}:
                techniques.append(name)
    tie = note.find("Tie")
    if tie is not None:
        if tie.get("origin") == "true":
            techniques.append("TieOrigin")
        if tie.get("destination") == "true":
            techniques.append("TieDestination")
    accent = note.findtext("Accent")
    if accent:
        techniques.append(f"Accent:{accent}")
    return techniques


def load_gpif(path: Path) -> ET.Element:
    try:
        with zipfile.ZipFile(path) as archive:
            return ET.fromstring(archive.read("Content/score.gpif"))
    except KeyError as exc:
        raise ValueError(f"{path} has no Content/score.gpif") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a modern ZIP-based .gp file") from exc


def extract(path: Path) -> dict[str, Any]:
    root = load_gpif(path)
    bars = id_map(root.find("Bars"))
    voices = id_map(root.find("Voices"))
    beats = id_map(root.find("Beats"))
    notes = id_map(root.find("Notes"))
    rhythms = id_map(root.find("Rhythms"))
    master_bars_node = root.find("MasterBars")
    tracks_node = root.find("Tracks")
    if master_bars_node is None or tracks_node is None:
        raise ValueError("GPIF lacks MasterBars or Tracks")
    master_bars = list(master_bars_node)
    tracks = list(tracks_node)

    track_info: list[dict[str, Any]] = []
    for track_index, track in enumerate(tracks):
        pitches = [int(v) for v in required_text(track, ".//Property[@name='Tuning']/Pitches").split()]
        capo_node = track.find(".//Property[@name='CapoFret']/Fret")
        capo = int(capo_node.text) if capo_node is not None and capo_node.text else 0
        track_info.append(
            {
                "index": track_index,
                "id": track.get("id"),
                "name": track.findtext("Name", default=f"Track {track_index + 1}"),
                "tuning_midi_low_to_high": pitches,
                "capo_fret": capo,
            }
        )

    events: list[dict[str, Any]] = []
    time_signatures: list[dict[str, int]] = []
    unsupported_structure: list[str] = []
    measure_start = Fraction(0)
    for measure_index, master_bar in enumerate(master_bars):
        time_text = required_text(master_bar, "Time")
        numerator, denominator = (int(v) for v in time_text.split("/"))
        if not time_signatures or (
            time_signatures[-1]["numerator"], time_signatures[-1]["denominator"]
        ) != (numerator, denominator):
            time_signatures.append(
                {
                    "measure": measure_index + 1,
                    "onset_beats": float(measure_start),
                    "numerator": numerator,
                    "denominator": denominator,
                }
            )
        structural_tags = {
            child.tag for child in master_bar
            if "repeat" in child.tag.lower() or "alternate" in child.tag.lower()
        }
        for tag in structural_tags:
            marker = f"playback_structure:{tag}"
            if marker not in unsupported_structure:
                unsupported_structure.append(marker)
        measure_duration = Fraction(numerator * 4, denominator)
        bar_refs = refs(master_bar, "Bars")

        for track_index, bar_id in enumerate(bar_refs):
            if track_index >= len(track_info):
                continue
            track = track_info[track_index]
            string_count = len(track["tuning_midi_low_to_high"])
            bar = bars[bar_id]
            for voice_index, voice_id in enumerate(refs(bar, "Voices")):
                onset = measure_start
                for beat_id in refs(voices[voice_id], "Beats"):
                    beat = beats[beat_id]
                    rhythm_ref = beat.find("Rhythm")
                    if rhythm_ref is None or rhythm_ref.get("ref") is None:
                        raise ValueError(f"Beat {beat_id} lacks a Rhythm reference")
                    duration = rhythm_beats(rhythms[rhythm_ref.get("ref")])
                    dynamic = beat.findtext("Dynamic", default="MF")

                    for note_id in refs(beat, "Notes"):
                        note = notes[note_id]
                        gpif_string = property_int(note, "String", "String")
                        fret = property_int(note, "Fret", "Fret")
                        midi = property_int(note, "Midi", "Number")
                        if not 0 <= gpif_string < string_count:
                            raise ValueError(f"Note {note_id} has invalid string index {gpif_string}")
                        expected_midi = track["tuning_midi_low_to_high"][gpif_string] + fret
                        events.append(
                            {
                                "track": track_index,
                                "track_name": track["name"],
                                "measure": measure_index + 1,
                                "voice": voice_index,
                                "beat_id": beat_id,
                                "note_id": note_id,
                                "onset_beats": float(onset),
                                "duration_beats": float(duration),
                                "gpif_string_index": gpif_string,
                                "tab_string": string_count - gpif_string,
                                "fret": fret,
                                "pitch": midi,
                                "expected_pitch": expected_midi,
                                "pitch_valid": midi == expected_midi,
                                "velocity": DYNAMIC_VELOCITY.get(dynamic, 79),
                                "dynamic": dynamic,
                                "techniques": note_techniques(note),
                            }
                        )
                    onset += duration
        measure_start += measure_duration

    invalid = [event["note_id"] for event in events if not event["pitch_valid"]]
    tempo = None
    tempo_automations: list[dict[str, Any]] = []
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if automation.findtext("Type") == "Tempo":
            value = (automation.findtext("Value") or "").split()
            if value:
                entry = {
                    "bar_index": int(automation.findtext("Bar", default="0")),
                    "position": float(automation.findtext("Position", default="0")),
                    "bpm": float(value[0]),
                    "linear": automation.findtext("Linear", default="false") == "true",
                }
                tempo_automations.append(entry)
                if tempo is None:
                    tempo = entry["bpm"]
    if len(tempo_automations) > 1 or any(
        entry["bar_index"] != 0 or entry["position"] != 0
        for entry in tempo_automations
    ):
        unsupported_structure.append("tempo_automation_after_start")
    if root.findall(".//GraceNotes"):
        unsupported_structure.append("grace_notes")

    return {
        "source": str(path),
        "gp_version": root.findtext("GPVersion"),
        "title": root.findtext("./Score/Title", default=""),
        "tempo_bpm_initial": tempo,
        "tempo_automations": tempo_automations,
        "time_signatures": time_signatures,
        "unsupported_structure": sorted(set(unsupported_structure)),
        "tracks": track_info,
        "validation": {
            "note_count": len(events),
            "pitch_mismatch_count": len(invalid),
            "pitch_mismatch_note_ids": invalid,
        },
        "notes": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    source = args.file.resolve()
    output = args.output or source.with_suffix(".notes.json")
    data = extract(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = data["validation"]
    print(f"Wrote: {output.resolve()}")
    print(f"Notes: {validation['note_count']}")
    print(f"Pitch mismatches: {validation['pitch_mismatch_count']}")


if __name__ == "__main__":
    main()
