#!/usr/bin/env python3
"""Fetch and audit MusicXML structure for opened PianoCoRe development only."""

import base64
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/p3_pianocore_p3r_manifest.json"
OUT = ROOT / "data/pianocore_p3r_development"
RESULT = ROOT / "results/p3_r_repeat_structure_audit.json"
API = "https://datasets-server.huggingface.co/rows"
JUMP_WORDS = re.compile(r"(?:\bD\.?\s*[CS]\.?\b|da\s+capo|dal\s+segno|to\s+coda|al\s+fine)", re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_row(session, expected):
    params = {"dataset": "SyMuPe/PianoCoRe", "config": "default",
              "split": expected["split"], "offset": expected["dataset_viewer_offset"], "length": 1}
    for attempt in range(5):
        response = session.get(API, params=params, timeout=90)
        if response.status_code == 200 and len(response.json().get("rows", [])) == 1:
            row = response.json()["rows"][0]["row"]
            if row["id"] != expected["id"]:
                raise AssertionError(f"offset identity mismatch: {row['id']} != {expected['id']}")
            return row
        if attempt == 4:
            raise RuntimeError(f"row fetch failed: {response.status_code} {response.text[:200]}")
        time.sleep(2 ** attempt)


def xml_from_bytes(data):
    """Return score XML bytes from plain XML or an MXL/ZIP payload."""
    if not data.startswith(b"PK"):
        return data, None
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".xml", ".musicxml"))]
        score_names = [name for name in names if not name.lower().endswith("meta-inf/container.xml")]
        if not score_names:
            raise ValueError("compressed MusicXML contains no score document")
        name = sorted(score_names, key=lambda value: (value.count("/"), len(value), value))[0]
        return archive.read(name), name


def local(tag):
    return tag.rsplit("}", 1)[-1]


def audit_xml(data):
    xml, member = xml_from_bytes(data)
    root = ET.fromstring(xml)
    counts = Counter()
    ending_numbers = set()
    jump_words = []
    measure_numbers = []
    for element in root.iter():
        tag = local(element.tag)
        if tag == "measure":
            counts["measures"] += 1
            if "number" in element.attrib:
                measure_numbers.append(element.attrib["number"])
        elif tag == "repeat":
            direction = element.attrib.get("direction", "unknown").lower()
            counts[f"repeat_{direction}"] += 1
            if direction == "backward" and element.attrib.get("times"):
                counts["repeat_backward_with_times"] += 1
        elif tag == "ending":
            counts[f"ending_{element.attrib.get('type', 'unknown').lower()}"] += 1
            ending_numbers.update(filter(None, re.split(r"\s*,\s*|\s+", element.attrib.get("number", ""))))
        elif tag in {"segno", "coda"}:
            counts[tag] += 1
        elif tag == "sound":
            for attr in ("dacapo", "dalsegno", "tocoda", "fine", "segno", "coda"):
                if element.attrib.get(attr) not in (None, "", "no", "false", "0"):
                    counts[f"sound_{attr}"] += 1
        elif tag == "words" and element.text and JUMP_WORDS.search(element.text):
            counts["jump_words"] += 1
            jump_words.append(element.text.strip()[:160])
    repeat_pair = counts["repeat_forward"] > 0 and counts["repeat_backward"] > 0
    ending = sum(value for key, value in counts.items() if key.startswith("ending_")) > 0
    jump = any(counts[key] for key in counts if key.startswith("sound_")) or counts["segno"] or counts["coda"] or counts["jump_words"]
    return {
        "xml_member": member, "xml_sha256": digest(xml), "root": local(root.tag),
        "counts": dict(sorted(counts.items())), "ending_numbers": sorted(ending_numbers),
        "jump_words": jump_words, "measure_number_examples": measure_numbers[:5],
        "has_repeat_pair": bool(repeat_pair), "has_ending": bool(ending),
        "has_jump": bool(jump), "structure_eligible": bool(repeat_pair or ending or jump),
    }


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["development"]
    audited = []
    with requests.Session() as session:
        for number, expected in enumerate(rows, 1):
            folder = OUT / expected["id"]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "score.mxl"
            if path.exists():
                payload = path.read_bytes()
                source = "cached"
            else:
                row = fetch_row(session, expected)
                encoded = row.get("score_xml_bytes")
                if not encoded:
                    audited.append({"id": expected["id"], "composer": expected["composer"],
                                    "score_dataset": expected["score_dataset"], "parse_ok": False,
                                    "error": "score_xml_bytes missing"})
                    continue
                payload = base64.b64decode(encoded)
                path.write_bytes(payload)
                source = "fetched"
            base = {"id": expected["id"], "composer": expected["composer"],
                    "score_dataset": expected["score_dataset"], "payload_bytes": len(payload),
                    "payload_sha256": digest(payload), "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "source": source}
            try:
                audited.append(base | {"parse_ok": True} | audit_xml(payload))
            except Exception as exc:
                audited.append(base | {"parse_ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{number}/{len(rows)}] {expected['id']} parse={audited[-1]['parse_ok']} eligible={audited[-1].get('structure_eligible', False)}")

    parse_ok = [row for row in audited if row["parse_ok"]]
    eligible = [row for row in parse_ok if row["structure_eligible"]]
    gates = {
        "parse_rate_ge_0_80": len(parse_ok) / len(rows) >= 0.80,
        "eligible_files_ge_10": len(eligible) >= 10,
        "eligible_composers_ge_5": len({row["composer"] for row in eligible}) >= 5,
        "eligible_score_datasets_ge_2": len({row["score_dataset"] for row in eligible}) >= 2,
    }
    result = {
        "scope": "Outcome-blind structural audit of opened PianoCoRe development XML only",
        "protocol": "P3_R_REPEAT_GRAPH_PROTOCOL.md", "manifest_sha256": digest(MANIFEST.read_bytes()),
        "requested": len(rows), "preserved_validation_requested": 0,
        "summary": {"parse_ok": len(parse_ok), "parse_rate": len(parse_ok) / len(rows),
                    "structure_eligible": len(eligible),
                    "eligible_composers": len({row["composer"] for row in eligible}),
                    "eligible_score_datasets": sorted({row["score_dataset"] for row in eligible}),
                    "repeat_pair": sum(row.get("has_repeat_pair", False) for row in parse_ok),
                    "ending": sum(row.get("has_ending", False) for row in parse_ok),
                    "jump": sum(row.get("has_jump", False) for row in parse_ok)},
        "gates": gates, "continue_to_graph": all(gates.values()), "rows": audited,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
