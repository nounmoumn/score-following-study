"""Freeze an outcome-blind Batik external cohort manifest."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/batik_plays_mozart"
OUT = ROOT / "results/p3_batik_external"
EXCLUDED = {"279": "development family", "331": "ASAP overlap", "332": "ASAP overlap"}
PATTERN = re.compile(r"kv(\d+)_(\d+)\.mid$", re.I)
MATCHED = re.compile(r"^snote\(([^,]+),.*?,(-?\d+\.\d+),(-?\d+\.\d+),\[[^]]*\]\)-note\([^,]+,(\d+),(\d+),")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_inversions(path):
    text = path.read_text(encoding="utf-8")
    units = int(re.search(r"info\(midiClockUnits,(\d+)\)", text).group(1))
    rate = int(re.search(r"info\(midiClockRate,(\d+)\)", text).group(1))
    by_beat = defaultdict(list)
    for line in text.splitlines():
        match = MATCHED.match(line)
        if match:
            by_beat[float(match.group(2))].append(int(match.group(5)) * rate / (units * 1_000_000))
    times = [sorted(values)[len(values) // 2] if len(values) % 2 else
             (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
             for _, values in sorted(by_beat.items())]
    differences = [b - a for a, b in zip(times, times[1:])]
    return sum(d <= 0 for d in differences), min(differences, default=float("inf"))


def main():
    rows = []
    for midi in sorted((DATA / "midi").glob("*.mid")):
        match = PATTERN.fullmatch(midi.name)
        if not match:
            raise RuntimeError(f"Unexpected filename: {midi.name}")
        sonata, movement = match.groups()
        stem = midi.stem.lower()
        score = DATA / "scores_edited" / f"{stem}.musicxml"
        alignment = DATA / "match" / f"{stem}.match"
        duration = float(pretty_midi.PrettyMIDI(str(midi)).get_end_time())
        reason = EXCLUDED.get(sonata)
        if reason is None and duration < 90:
            reason = "performance shorter than frozen 90 seconds"
        if not score.exists() or not alignment.exists():
            reason = "missing score or match"
        inversions, minimum_step = clock_inversions(alignment) if alignment.exists() else (None, None)
        if reason is None and inversions:
            reason = "non-monotone matched performance clock"
        rows.append({
            "sonata": f"K.{sonata}", "movement": int(movement),
            "performance": str(midi.relative_to(ROOT)).replace("\\", "/"),
            "score": str(score.relative_to(ROOT)).replace("\\", "/"),
            "match": str(alignment.relative_to(ROOT)).replace("\\", "/"),
            "duration_sec": duration, "eligible": reason is None,
            "exclusion_reason": reason,
            "clock_nonpositive_steps": inversions, "clock_min_step_sec": minimum_step,
            "sha256": {"performance": sha256(midi), "score": sha256(score), "match": sha256(alignment)}
            if score.exists() and alignment.exists() else None,
        })
    result = {
        "scope": "outcome-blind external cohort freeze; no tracker output read",
        "source_commit": "9c5f7004c5ab1f15d64e959d011dbdf3b7863c3e",
        "rules": {"minimum_duration_sec": 90, "excluded_sonatas": EXCLUDED},
        "total_movements": len(rows), "eligible_movements": sum(r["eligible"] for r in rows),
        "eligible_sonatas": sorted({r["sonata"] for r in rows if r["eligible"]}),
        "rows": rows,
        "protocol_sha256": sha256(ROOT / "P3_BATIK_EXTERNAL_PROTOCOL.md"),
        "script_sha256": sha256(Path(__file__)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
