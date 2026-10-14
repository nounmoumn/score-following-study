"""Validate rendered WAV files and write audio_qa.json/csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from prepare_reaper_batch import validate_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/guitar_dataset"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        wav = args.dataset / "renders" / f"{item['id']}.wav"
        try:
            qa = validate_wav(wav)
            qa["id"] = item["id"]
            qa["score_status"] = item["status"]
            qa["audio_status"] = "valid"
            qa["error"] = None
        except Exception as exc:
            qa = {
                "id": item["id"],
                "path": str(wav.resolve()),
                "score_status": item["status"],
                "audio_status": "missing_or_invalid",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(qa)
        print(f"{qa['audio_status']}: {item['id']}")

    json_path = args.dataset / "audio_qa.json"
    csv_path = args.dataset / "audio_qa.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"QA manifest: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
