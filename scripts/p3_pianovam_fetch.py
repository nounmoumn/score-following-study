#!/usr/bin/env python3
"""Fetch frozen PianoVAM direct MIDI and catalog-matched score files only."""

import base64
import hashlib
import json
import time
from pathlib import Path

import pretty_midi
import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_pianovam_score_manifest.json"
OUT = ROOT / "data/pianovam_p3_natural"
REPORT = ROOT / "results/p3_pianovam_acquisition.json"
HF = "https://huggingface.co/datasets/PianoVAM/PianoVAM_v1/resolve"
ROWS = "https://datasets-server.huggingface.co/rows"


def digest(data): return hashlib.sha256(data).hexdigest()


def get(session, url, **kwargs):
    for attempt in range(5):
        response = session.get(url, timeout=90, **kwargs)
        if response.status_code == 200: return response
        if attempt == 4: raise RuntimeError(f"fetch failed {response.status_code}: {url}")
        time.sleep(2 ** attempt)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [row for row in manifest["rows"] if row["catalog_match"]]
    OUT.mkdir(parents=True, exist_ok=True); acquired = []; score_cache = {}
    with requests.Session() as session:
        for number, row in enumerate(rows, 1):
            perf_url = f"{HF}/{manifest['source_commit']}/{row['midi_repo_path']}?download=true"
            performance = get(session, perf_url).content
            if not performance.startswith(b"MThd"): raise AssertionError(f"bad MIDI {row['record_time']}")
            pfolder = OUT / "performances"; pfolder.mkdir(exist_ok=True)
            ppath = pfolder / f"{row['record_time']}.mid"; ppath.write_bytes(performance)
            pm = pretty_midi.PrettyMIDI(str(ppath)); notes = [note for inst in pm.instruments for note in inst.notes]
            score = row["score"]; key = str(row["rule_index"])
            if key not in score_cache:
                params = {"dataset": "SyMuPe/PianoCoRe", "config": "default", "split": score["split"],
                          "offset": score["dataset_viewer_offset"], "length": 1}
                payload = get(session, ROWS, params=params).json()["rows"][0]["row"]
                if payload["score_id"] != score["score_id"]: raise AssertionError("score offset identity mismatch")
                smidi = base64.b64decode(payload["score_midi_bytes"]); sxml = base64.b64decode(payload["score_xml_bytes"])
                if not smidi.startswith(b"MThd") or not sxml: raise AssertionError("invalid score payload")
                sfolder = OUT / "scores" / f"rule_{int(key):02d}"; sfolder.mkdir(parents=True, exist_ok=True)
                (sfolder / "score.mid").write_bytes(smidi); (sfolder / "score.mxl").write_bytes(sxml)
                score_cache[key] = {"score_midi": smidi, "score_xml": sxml,
                                    "midi_path": str((sfolder/"score.mid").relative_to(ROOT)).replace("\\", "/"),
                                    "xml_path": str((sfolder/"score.mxl").relative_to(ROOT)).replace("\\", "/")}
            cached = score_cache[key]
            if digest(performance) == digest(cached["score_midi"]): raise AssertionError("performance equals score")
            acquired.append({**{k: row[k] for k in ("record_time","split","composer","piece","performer","rule_index")},
                             "performance_path": str(ppath.relative_to(ROOT)).replace("\\", "/"),
                             "performance_bytes": len(performance), "performance_sha256": digest(performance),
                             "performance_duration_sec": float(pm.get_end_time()), "performance_notes": len(notes),
                             "score_id": score["score_id"], "score_dataset": score["score_dataset"],
                             "score_composer": score["composer"],
                             "score_midi_path": cached["midi_path"], "score_xml_path": cached["xml_path"],
                             "score_midi_sha256": digest(cached["score_midi"]),
                             "score_xml_sha256": digest(cached["score_xml"])})
            print(f"[{number}/{len(rows)}] {row['record_time']} {row['piece']}")
    result = {"scope": "PianoVAM direct-MIDI and third-party score acquisition; no tracker outcomes",
              "manifest_sha256": digest(MANIFEST.read_bytes()), "recordings": len(acquired),
              "work_rules": len(set(row["rule_index"] for row in acquired)),
              "composers": len(set(row["score_composer"] for row in acquired)),
              "all_performance_score_distinct": True, "validation_eligible": False, "rows": acquired}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
