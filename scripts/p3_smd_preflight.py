"""Inventory SMD direct MIDI files without computing score-following outcomes."""

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

import pretty_midi
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi"
OUT = ROOT / "results/p3_external_metadata/smd_preflight.json"
RAW = ROOT / "data/smd_direct_midi"


def fetch(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return response.read()


def main():
    page = BeautifulSoup(fetch(SOURCE), "html.parser")
    links = [(link.get_text(strip=True), urljoin(SOURCE, link["href"]))
             for link in page.select("a[href]") if link.get_text(strip=True).lower() == "mid"]
    if len(links) != 50:
        raise RuntimeError(f"Expected 50 official MIDI links, found {len(links)}")
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (_, url) in enumerate(links, 1):
        filename = url.rsplit("/", 1)[-1]
        if not filename.endswith(".mid") or not re.fullmatch(r"[A-Za-z0-9_-]+\.mid", filename):
            raise RuntimeError(f"Unexpected MIDI filename: {filename}")
        path = RAW / filename
        if not path.exists():
            path.write_bytes(fetch(url))
        midi = pretty_midi.PrettyMIDI(str(path))
        parts = filename.removesuffix(".mid").split("_")
        rows.append({"index": index, "filename": filename, "url": url,
                     "composer": parts[0], "work_code": parts[1],
                     "family_code": re.sub(r"-\d\d$", "", parts[1]),
                     "duration_sec": float(midi.get_end_time()),
                     "note_count": int(sum(len(inst.notes) for inst in midi.instruments)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    summary = {"source": SOURCE, "rows": rows,
               "total": len(rows), "duration_ge_90s": sum(row["duration_sec"] >= 90 for row in rows),
               "composer_counts": dict(Counter(row["composer"] for row in rows)),
               "family_count": len({(row["composer"], row["family_code"]) for row in rows}),
               "caveat": "Direct performance MIDI only; independent scores and score-performance ground truth absent. No tracker or outcome evaluated."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
