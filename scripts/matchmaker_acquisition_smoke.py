"""Pinned Matchmaker acquisition smoke test on opened development case fd000."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

import mido
import numpy as np
import partitura as pt


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output" / "ambiguity_map_fresh_cohort" / "manifest.csv"
PROTOCOL = ROOT / "MATCHMAKER_AMBIGUITY_RESCUE_PROTOCOL.md"
OUT = ROOT / "results" / "matchmaker_acquisition_smoke"
METHODS = ("arzt", "dixon", "outerhmm", "hmm", "pthmm")
LONG_PREFIX_SEC = 30.0
SHORT_PREFIX_SEC = 15.0
EXPECTED_COMMIT = "40fd6e7602bd9e596cea1beafdcee4853b9f2c98"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def development_case(case_id: str = "fd000") -> dict[str, str]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["case_id"] == case_id]
    if len(matches) != 1 or matches[0]["role"] != "development":
        raise ValueError(f"Expected one opened development case {case_id}")
    return matches[0]


def seconds_to_tick(mid: mido.MidiFile, cutoff_sec: float) -> int:
    """Return the final global tick not later than cutoff_sec."""
    tempo = 500_000
    last_tick = 0
    elapsed = 0.0
    for message in mido.merge_tracks(mid.tracks):
        next_tick = last_tick + int(message.time)
        next_elapsed = elapsed + mido.tick2second(
            next_tick - last_tick, mid.ticks_per_beat, tempo
        )
        if next_elapsed > cutoff_sec:
            remaining = cutoff_sec - elapsed
            return last_tick + int(
                np.floor(mido.second2tick(remaining, mid.ticks_per_beat, tempo))
            )
        elapsed = next_elapsed
        last_tick = next_tick
        if message.type == "set_tempo":
            tempo = message.tempo
    return last_tick


def write_exact_tick_prefix(source: Path, target: Path, cutoff_sec: float) -> int:
    """Copy every per-track event through a shared global tick cutoff."""
    original = mido.MidiFile(source)
    cutoff_tick = seconds_to_tick(original, cutoff_sec)
    prefix = mido.MidiFile(type=original.type, ticks_per_beat=original.ticks_per_beat)
    for track in original.tracks:
        new_track = mido.MidiTrack()
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if absolute_tick > cutoff_tick:
                break
            if message.type != "end_of_track":
                new_track.append(message.copy())
        new_track.append(mido.MetaMessage("end_of_track", time=0))
        prefix.tracks.append(new_track)
    prefix.save(target)
    return cutoff_tick


def score_clock(score_midi: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build a GT-free score-beat to score-MIDI-second lookup."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score_notes = pt.load_score(str(score_midi)).note_array()
        midi_notes = pt.load_performance_midi(str(score_midi)).note_array()
    if len(score_notes) != len(midi_notes):
        raise ValueError("score/performance views have different note counts")
    if not np.array_equal(score_notes["pitch"], midi_notes["pitch"]):
        raise ValueError("score/performance views have different pitch order")
    tick_residual = np.asarray(score_notes["onset_div"], float) - np.asarray(
        midi_notes["onset_tick"], float
    )
    if np.max(np.abs(tick_residual), initial=0.0) > 1e-9:
        raise ValueError("score divisions do not match MIDI ticks")

    beat = np.asarray(score_notes["onset_beat"], float)
    second = np.asarray(midi_notes["onset_sec"], float)
    unique_beats, inverse = np.unique(beat, return_inverse=True)
    unique_seconds = np.empty_like(unique_beats)
    max_spread = 0.0
    for index in range(len(unique_beats)):
        values = second[inverse == index]
        unique_seconds[index] = float(np.median(values))
        max_spread = max(max_spread, float(np.ptp(values)))
    if max_spread > 1e-7 or np.any(np.diff(unique_seconds) < -1e-9):
        raise ValueError("score clock is not single-valued and monotone")
    audit = {
        "note_count": int(len(score_notes)),
        "unique_onsets": int(len(unique_beats)),
        "max_tick_residual": float(np.max(np.abs(tick_residual), initial=0.0)),
        "max_same_beat_second_spread": max_spread,
        "beat_range": [float(unique_beats[0]), float(unique_beats[-1])],
        "second_range": [float(unique_seconds[0]), float(unique_seconds[-1])],
    }
    return unique_beats, unique_seconds, audit


def run_tracker(score: Path, performance: Path, method: str) -> np.ndarray:
    from matchmaker import Matchmaker

    tracker = Matchmaker(
        score_file=score,
        performance_file=performance,
        input_type="midi",
        method=method,
        wait=False,
        unfold_score=False,
    )
    list(tracker.run(verbose=False))
    return np.asarray(tracker.score_follower.alignment_path, dtype=float)


def prefix_agreement(long_path: np.ndarray, short_path: np.ndarray) -> dict[str, object]:
    # Exclude a conservative 50 ms boundary region where a window can be flushed.
    safe = SHORT_PREFIX_SEC - 0.05
    left = long_path[:, long_path[0] <= safe]
    right = short_path[:, short_path[0] <= safe]
    same_shape = left.shape == right.shape
    max_abs = (
        float(np.max(np.abs(left - right), initial=0.0)) if same_shape else None
    )
    return {
        "safe_cutoff_sec": safe,
        "long_points": int(left.shape[1]),
        "short_points": int(right.shape[1]),
        "same_shape": same_shape,
        "max_abs_difference": max_abs,
        "pass": bool(same_shape and max_abs is not None and max_abs <= 1e-9),
    }


def validate_path(path: np.ndarray, beat_min: float, beat_max: float) -> dict[str, object]:
    shape_ok = path.ndim == 2 and path.shape[0] == 2 and path.shape[1] >= 20
    finite = bool(shape_ok and np.isfinite(path).all())
    perf_monotone = bool(finite and np.all(np.diff(path[0]) >= -1e-12))
    beat_in_bounds = bool(
        finite
        and np.all(path[1] >= beat_min - 1e-7)
        and np.all(path[1] <= beat_max + 1e-7)
    )
    return {
        "shape": list(path.shape),
        "finite": finite,
        "performance_time_nondecreasing": perf_monotone,
        "score_beat_in_bounds": beat_in_bounds,
        "performance_time_range": (
            [float(path[0].min()), float(path[0].max())] if finite else None
        ),
        "score_beat_range": (
            [float(path[1].min()), float(path[1].max())] if finite else None
        ),
        "pass": bool(shape_ok and finite and perf_monotone and beat_in_bounds),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    OUT.mkdir(parents=True)
    row = development_case()
    score = ROOT / "asap_dataset" / row["score"]
    performance = ROOT / "asap_dataset" / row["performance"]
    long_midi = OUT / "fd000_prefix_30s.mid"
    short_midi = OUT / "fd000_prefix_15s.mid"
    long_tick = write_exact_tick_prefix(performance, long_midi, LONG_PREFIX_SEC)
    short_tick = write_exact_tick_prefix(performance, short_midi, SHORT_PREFIX_SEC)
    beats, seconds, clock_audit = score_clock(score)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "matchmaker"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Unexpected Matchmaker commit {commit}")

    frozen = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in (Path(__file__), PROTOCOL, MANIFEST, score, performance)
    }
    write_json(OUT / "freeze.json", frozen)
    methods: dict[str, object] = {}
    for method in METHODS:
        print(f"[{method}] long prefix", flush=True)
        try:
            long_path = run_tracker(score, long_midi, method)
            print(f"[{method}] short prefix", flush=True)
            short_path = run_tracker(score, short_midi, method)
            path_check = validate_path(long_path, float(beats[0]), float(beats[-1]))
            prefix_check = prefix_agreement(long_path, short_path)
            converted = np.interp(long_path[1], beats, seconds)
            conversion_finite = bool(np.isfinite(converted).all())
            passed = bool(path_check["pass"] and prefix_check["pass"] and conversion_finite)
            methods[method] = {
                "status": "PASS" if passed else "FAIL",
                "path": path_check,
                "prefix": prefix_check,
                "score_second_conversion_finite": conversion_finite,
            }
        except Exception as exc:
            methods[method] = {
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_json(OUT / "partial.json", methods)

    if any(sha256(ROOT / relative) != digest for relative, digest in frozen.items()):
        raise RuntimeError("Frozen input changed during smoke run")
    passing = [name for name, result in methods.items() if result["status"] == "PASS"]
    result = {
        "status": "PASS" if len(passing) >= 2 else "STOP",
        "case_id": row["case_id"],
        "matchmaker_commit": commit,
        "python": sys.version,
        "prefix": {
            "long_seconds": LONG_PREFIX_SEC,
            "short_seconds": SHORT_PREFIX_SEC,
            "long_cutoff_tick": long_tick,
            "short_cutoff_tick": short_tick,
        },
        "score_clock": clock_audit,
        "methods": methods,
        "passing_methods": passing,
        "development_authorized": len(passing) >= 2,
        "lockbox_authorized": False,
    }
    write_json(OUT / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
