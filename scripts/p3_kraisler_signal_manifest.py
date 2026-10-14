"""Freeze KRAISLER eligibility using files and clocks only."""

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/kraisler_symbolic"
OUT = ROOT / "results/p3_kraisler_signal"
PROTOCOL = ROOT / "P3_KRAISLER_SIGNAL_PROTOCOL.md"
MATCHED = re.compile(r"^snote\(([^,]+),.*?,(-?\d+\.\d+),(-?\d+\.\d+),\[[^]]*\]\)-note\([^,]+,(\d+),(\d+),")


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def clock(path):
    text = path.read_text(encoding="utf-8")
    units = int(re.search(r"info\(midiClockUnits,(\d+)\)", text).group(1))
    rate = int(re.search(r"info\(midiClockRate,(\d+)\)", text).group(1))
    by_beat = defaultdict(list)
    for line in text.splitlines():
        m = MATCHED.match(line)
        if m: by_beat[float(m.group(2))].append(int(m.group(5)) * rate / (units * 1_000_000))
    beats = np.array(sorted(by_beat)); times = np.array([np.median(by_beat[b]) for b in beats])
    differences = np.diff(times)
    return len(beats), int(np.sum(differences <= 0)), float(differences.min())


def main():
    rows = []
    with (DATA / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    for item in metadata:
        performance = DATA / item["performance_midi_PF"]
        score = DATA / item["score_musicxml_PF"]
        match = DATA / item["alignment_match"]
        duration = float(pretty_midi.PrettyMIDI(str(performance)).get_end_time())
        anchors, inversions, minimum_step = clock(match)
        root = ET.parse(score).getroot()
        title = (root.findtext("./work/work-title") or root.findtext("./movement-title") or item["track"]).strip()
        composer = (root.findtext("./identification/creator") or "unknown").strip()
        reason = None
        if duration < 45: reason = "shorter than 45 seconds"
        elif inversions: reason = "non-monotone matched performance clock"
        rows.append({"track": item["track"], "title": title, "composer": composer,
                     "performance": str(performance.relative_to(ROOT)).replace("\\", "/"),
                     "score": str(score.relative_to(ROOT)).replace("\\", "/"),
                     "match": str(match.relative_to(ROOT)).replace("\\", "/"),
                     "duration_sec": duration, "anchor_count": anchors,
                     "clock_nonpositive_steps": inversions, "clock_min_step_sec": minimum_step,
                     "eligible": reason is None, "exclusion_reason": reason,
                     "sha256": {"performance": sha256(performance), "score": sha256(score), "match": sha256(match)}})
    result = {"scope": "outcome-blind KRAISLER signal manifest", "total": len(rows),
              "eligible": sum(r["eligible"] for r in rows), "rows": rows,
              "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
