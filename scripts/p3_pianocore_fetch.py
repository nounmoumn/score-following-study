#!/usr/bin/env python3
"""Fetch only the frozen PianoCoRe P3-R development rows via Dataset Viewer."""

import base64
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_pianocore_p3r_manifest.json"
OUT = ROOT / "data/pianocore_p3r_development"
REPORT = ROOT / "results/p3_pianocore_p3r_acquisition.json"
API = "https://datasets-server.huggingface.co/rows"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_row(session, split, offset):
    params = {"dataset": "SyMuPe/PianoCoRe", "config": "default", "split": split,
              "offset": offset, "length": 1}
    for attempt in range(5):
        response = session.get(API, params=params, timeout=90)
        if response.status_code == 200:
            payload = response.json()
            if len(payload.get("rows", [])) == 1:
                return payload["rows"][0]["row"]
        if attempt == 4:
            raise RuntimeError(f"row fetch failed: {response.status_code} {response.text[:200]}")
        time.sleep(2 ** attempt)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["development"]
    OUT.mkdir(parents=True, exist_ok=True)
    acquired = []
    with requests.Session() as session:
        for number, expected in enumerate(rows, 1):
            row = fetch_row(session, expected["split"], expected["dataset_viewer_offset"])
            if row["id"] != expected["id"]:
                raise AssertionError(f"offset identity mismatch: {row['id']} != {expected['id']}")
            score = base64.b64decode(row["score_midi_bytes"])
            performance = base64.b64decode(row["performance_midi_bytes"])
            alignment = base64.b64decode(row["raw_alignment_bytes"])
            if not score.startswith(b"MThd") or not performance.startswith(b"MThd"):
                raise AssertionError(f"invalid MIDI header: {row['id']}")
            if not alignment.startswith(b"PK"):
                raise AssertionError(f"invalid alignment NPZ: {row['id']}")
            if digest(score) == digest(performance):
                raise AssertionError(f"score/performance bytes identical: {row['id']}")
            folder = OUT / row["id"]
            folder.mkdir(exist_ok=True)
            (folder / "score.mid").write_bytes(score)
            (folder / "performance.mid").write_bytes(performance)
            (folder / "alignment.npz").write_bytes(alignment)
            acquired.append({
                "id": row["id"], "composer": row["composer"], "composition": row["composition"],
                "movement": row["movement"], "performance_dataset": row["performance_dataset"],
                "score_dataset": row["score_dataset"], "dataset_viewer_offset": expected["dataset_viewer_offset"],
                "score_bytes": len(score), "performance_bytes": len(performance),
                "alignment_bytes": len(alignment), "score_sha256": digest(score),
                "performance_sha256": digest(performance), "alignment_sha256": digest(alignment),
                "score_path": str((folder / "score.mid").relative_to(ROOT)).replace("\\", "/"),
                "performance_path": str((folder / "performance.mid").relative_to(ROOT)).replace("\\", "/"),
                "alignment_path": str((folder / "alignment.npz").relative_to(ROOT)).replace("\\", "/"),
            })
            print(f"[{number}/{len(rows)}] {row['id']} {row['composer']}")
    result = {
        "scope": "PianoCoRe P3-R development acquisition only; no outcomes computed",
        "manifest_sha256": digest(MANIFEST.read_bytes()), "requested": len(rows),
        "acquired": len(acquired), "all_score_performance_distinct": True,
        "preserved_validation_requested": 0, "rows": acquired,
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
