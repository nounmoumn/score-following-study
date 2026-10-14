"""Create per-item REAPER projects from one configured Stratus template."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import mido
import numpy as np
import soundfile as sf


MIDI_FILE_RE = re.compile(r'(?m)^(\s*FILE\s+")[^"]+\.mid(".*)$', re.IGNORECASE)


def replace_value(text: str, key: str, value: str) -> str:
    escaped = value.replace("\\", "/")
    pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s+.*$")
    replacement = f'  {key} "{escaped}"'
    if pattern.search(text):
        return pattern.sub(lambda _: replacement, text, count=1)
    closing = text.rfind(">")
    if closing < 0:
        raise ValueError("Invalid REAPER project: missing closing '>'")
    return text[:closing] + f'  {key} "{escaped}"\n' + text[closing:]


def midi_event_lines(midi_path: Path, indent: str) -> list[str]:
    midi = mido.MidiFile(midi_path)
    absolute_tick = 0
    previous_channel_tick = 0
    lines: list[str] = []
    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += message.time
        if message.is_meta or message.type == "sysex":
            continue
        delta = absolute_tick - previous_channel_tick
        payload = " ".join(f"{byte:02x}" for byte in message.bytes())
        lines.append(f"{indent}E {delta} {payload}")
        previous_channel_tick = absolute_tick
    lines.append(f"{indent}E 0 b0 7b 00")
    return lines


def replace_embedded_midi(template: str, midi_path: Path) -> str:
    lines = template.splitlines()
    source_starts = [i for i, line in enumerate(lines) if line.strip() == "<SOURCE MIDI"]
    if len(source_starts) != 1:
        raise ValueError(f"Template must contain exactly one embedded MIDI source; found {len(source_starts)}")
    start = source_starts[0]
    source_indent = len(lines[start]) - len(lines[start].lstrip())
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip() == ">" and len(lines[i]) - len(lines[i].lstrip()) == source_indent),
        None,
    )
    if end is None:
        raise ValueError("Embedded MIDI source has no closing marker")
    event_indexes = [i for i in range(start + 1, end) if lines[i].lstrip().startswith("E ")]
    if not event_indexes:
        raise ValueError("Embedded MIDI source contains no event lines")
    insertion = event_indexes[0]
    event_indent = lines[insertion][:-len(lines[insertion].lstrip())]
    retained = [line for i, line in enumerate(lines[start + 1:end], start + 1) if i not in set(event_indexes)]
    relative_insertion = sum(1 for i in range(start + 1, insertion) if i not in set(event_indexes))
    replacement = retained[:relative_insertion] + midi_event_lines(midi_path, event_indent) + retained[relative_insertion:]
    return "\n".join(lines[:start + 1] + replacement + lines[end:]) + "\n"


def make_project(template: str, midi_path: Path, wav_path: Path, tempo: float) -> str:
    midi_value = str(midi_path.resolve()).replace("\\", "/")
    matches = list(MIDI_FILE_RE.finditer(template))
    if len(matches) == 1:
        project = MIDI_FILE_RE.sub(
            lambda match: match.group(1) + midi_value + match.group(2), template, count=1
        )
    elif len(matches) == 0:
        project = replace_embedded_midi(template, midi_path)
    else:
        raise ValueError(f"Template has multiple MIDI FILE sources: {len(matches)}")

    midi = mido.MidiFile(midi_path)
    project = re.sub(
        r"(?m)^(\s*)TEMPO\s+[^\r\n]+$",
        lambda match: f"{match.group(1)}TEMPO {tempo:g} 4 4 0",
        project,
        count=1,
    )
    project = re.sub(
        r"(?m)^(\s*)LENGTH\s+[^\r\n]+$",
        lambda match: f"{match.group(1)}LENGTH {midi.length:.9f}",
        project,
        count=1,
    )
    project = replace_value(project, "RENDER_FILE", str(wav_path.parent.resolve()))
    project = replace_value(project, "RENDER_PATTERN", wav_path.stem)
    return project


def find_reaper() -> Path:
    candidates = [
        Path(r"C:\Program Files\REAPER (x64)\reaper.exe"),
        Path(r"C:\Program Files\REAPER\reaper.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("REAPER executable not found; pass --reaper-exe")


def validate_wav(path: Path) -> dict[str, float | int | str]:
    if not path.is_file():
        raise RuntimeError(f"REAPER did not create {path}")
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError(f"Invalid rendered WAV: {path}")
    peak = 0.0
    sum_squares = 0.0
    sample_count = 0
    non_silent = 0
    threshold = 10 ** (-60 / 20)
    with sf.SoundFile(path) as audio:
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            absolute = np.abs(block)
            peak = max(peak, float(absolute.max(initial=0.0)))
            sum_squares += float(np.square(block, dtype=np.float64).sum())
            sample_count += block.size
            non_silent += int((absolute > threshold).sum())
    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
    if peak == 0.0:
        raise RuntimeError(f"Rendered WAV is silent: {path}")
    return {
        "path": str(path.resolve()),
        "frames": info.frames,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": info.duration,
        "subtype": info.subtype,
        "peak": peak,
        "rms": rms,
        "non_silent_fraction_above_minus_60db": non_silent / sample_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="RPP with Kontakt + Stratus and one MIDI item")
    parser.add_argument("--dataset", type=Path, default=Path("data/guitar_dataset"))
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--only-id", action="append", default=[], help="Render only this manifest id")
    parser.add_argument("--render", action="store_true", help="Run REAPER headless rendering")
    parser.add_argument("--reaper-exe", type=Path)
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8", errors="replace")
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    projects_dir = args.dataset / "reaper_projects"
    renders_dir = args.dataset / "renders"
    projects_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    eligible = {"ready_exact_fingering", "ready_with_string_remap"}
    if args.include_partial:
        eligible.add("partial_out_of_range")

    projects: list[Path] = []
    for row in manifest:
        if args.only_id and not any(
            row["id"] == selector or row["id"].endswith(selector)
            for selector in args.only_id
        ):
            continue
        if row["status"] not in eligible:
            print(f"SKIP {row['id']}: {row['status']}")
            continue
        midi_path = args.dataset / "items" / row["id"] / "shreddage.mid"
        annotation_path = args.dataset / "items" / row["id"] / "notes.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        wav_path = renders_dir / f"{row['id']}.wav"
        project_path = projects_dir / f"{row['id']}.rpp"
        tempo = float(annotation.get("tempo_bpm_initial") or 120.0)
        project_path.write_text(
            make_project(template, midi_path, wav_path, tempo), encoding="utf-8"
        )
        projects.append(project_path)
        print(f"PROJECT {project_path}")

    print(f"Prepared projects: {len(projects)}")
    if args.render:
        reaper = args.reaper_exe or find_reaper()
        for index, project in enumerate(projects, 1):
            print(f"RENDER [{index}/{len(projects)}] {project.name}")
            result = subprocess.run(
                [str(reaper), "-renderproject", str(project.resolve())], check=False
            )
            wav_path = renders_dir / f"{project.stem}.wav"
            qa = validate_wav(wav_path)
            qa["reaper_exit_code"] = result.returncode
            qa_path = wav_path.with_suffix(".qa.json")
            qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
            print(
                f"OK {wav_path.name}: {qa['duration_seconds']:.3f}s, "
                f"{qa['samplerate']} Hz, {qa['channels']} ch "
                f"(REAPER exit {result.returncode})"
            )


if __name__ == "__main__":
    main()
