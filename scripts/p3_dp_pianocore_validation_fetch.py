#!/usr/bin/env python3
"""Fetch exactly the frozen 17-composer PianoCoRe P3-DP validation split."""

import base64
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_pianocore_p3r_manifest.json"
OUT = ROOT / "data/pianocore_p3dp_validation"
REPORT = ROOT / "results/p3_dp_pianocore_validation_acquisition.json"
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    expected_rows = manifest["preserved_validation"]
    if len(expected_rows) != 17:
        raise AssertionError("preserved split size changed")
    OUT.mkdir(parents=True, exist_ok=True)
    acquired = []
    with requests.Session() as session:
        for number, expected in enumerate(expected_rows, 1):
            row = fetch_row(session, expected["split"], expected["dataset_viewer_offset"])
            if row["id"] != expected["id"]:
                raise AssertionError("dataset-viewer identity mismatch")
            score = base64.b64decode(row["score_midi_bytes"])
            performance = base64.b64decode(row["performance_midi_bytes"])
            alignment = base64.b64decode(row["raw_alignment_bytes"])
            if not score.startswith(b"MThd") or not performance.startswith(b"MThd") or not alignment.startswith(b"PK"):
                raise AssertionError("invalid acquired file header")
            if digest(score) == digest(performance):
                raise AssertionError("score and performance are identical")
            folder = OUT / row["id"]
            folder.mkdir(exist_ok=True)
            (folder / "score.mid").write_bytes(score)
            (folder / "performance.mid").write_bytes(performance)
            (folder / "alignment.npz").write_bytes(alignment)
            acquired.append({
                "id": row["id"], "composer": row["composer"], "composition": row["composition"],
                "movement": row["movement"], "performance_dataset": row["performance_dataset"],
                "score_dataset": row["score_dataset"], "split": expected["split"],
                "dataset_viewer_offset": expected["dataset_viewer_offset"],
                "score_sha256": digest(score), "performance_sha256": digest(performance),
                "alignment_sha256": digest(alignment), "score_performance_distinct": digest(score) != digest(performance),
                "score_path": str((folder / "score.mid").relative_to(ROOT)).replace("\\", "/"),
                "performance_path": str((folder / "performance.mid").relative_to(ROOT)).replace("\\", "/"),
                "alignment_path": str((folder / "alignment.npz").relative_to(ROOT)).replace("\\", "/"),
            })
            print(f"[{number}/{len(expected_rows)}] {row['id']}")
    result = {
        "scope": "One-shot acquisition of frozen P3-DP preserved validation split",
        "manifest_sha256": digest(MANIFEST.read_bytes()), "requested": len(expected_rows),
        "acquired": len(acquired), "unique_composers": len({r["composer"] for r in acquired}),
        "all_score_performance_distinct": all(r["score_performance_distinct"] for r in acquired),
        "rows": acquired,
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

