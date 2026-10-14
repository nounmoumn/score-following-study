"""Inspect the KRAISLER archive index and metadata without fetching its audio."""

import csv
import hashlib
import io
import json
import struct
import urllib.request
import urllib.error
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
import time

import pretty_midi

RECORD = "https://zenodo.org/api/records/21082251"
OUT = Path(__file__).resolve().parents[1] / "results/p3_external_metadata/kraisler_preflight.json"


def get_range(url, start, end):
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                if response.status != 206 or len(data) != end - start + 1:
                    raise RuntimeError(f"range unsupported: status={response.status}, size={len(data)}")
                return data
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    archive = next(f for f in record["files"] if f["key"] == "KRAISLER.zip")
    url, size = archive["links"]["self"], archive["size"]
    tail = get_range(url, max(0, size - 65557), size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise RuntimeError("ZIP end-of-central-directory not found")
    _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack_from("<IHHHHIIH", tail, pos)
    central = get_range(url, cd_offset, cd_offset + cd_size - 1)
    entries = []
    stream = io.BytesIO(central)
    for _ in range(count):
        header = stream.read(46)
        if header[:4] != b"PK\x01\x02":
            raise RuntimeError("invalid ZIP central directory")
        vals = struct.unpack("<IHHHHHHIIIHHHHHII", header)
        compressed_size, uncompressed_size = vals[8], vals[9]
        name_len, extra_len, comment_len = vals[10:13]
        name = stream.read(name_len).decode("utf-8", errors="replace")
        stream.seek(extra_len + comment_len, 1)
        entries.append((name, compressed_size, uncompressed_size, vals[16], vals[4]))
    by_name = {entry[0]: entry for entry in entries}
    def extract(entry):
        _, compressed_size, _, offset, method = entry
        local = get_range(url, offset, offset + 29)
        _, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", local)
        payload_offset = offset + 30 + name_len + extra_len
        payload = get_range(url, payload_offset, payload_offset + compressed_size - 1)
        if method not in (0, 8):
            raise RuntimeError(f"unsupported compression method {method}")
        return zlib.decompress(payload, -15) if method == 8 else payload
    metadata = [entry for entry in entries if entry[0].endswith("metadata.csv")]
    if len(metadata) != 1:
        raise RuntimeError("expected one metadata.csv")
    raw = extract(metadata[0])
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    tracks = []
    for row in rows:
        midi = extract(by_name["KRAISLER/" + row["performance_midi_PF"]])
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
        track = {"track": row["track"], "duration_sec": pm.get_end_time(),
                 "midi_sha256": hashlib.sha256(midi).hexdigest(),
                 "paired_score_path": row["score_musicxml_PF"],
                 "paired_match_path": row["alignment_match"]}
        if track["duration_sec"] >= 90:
            score = extract(by_name["KRAISLER/" + row["score_musicxml_PF"]])
            root = ET.fromstring(score)
            track["title"] = (root.findtext("./work/work-title") or root.findtext("./movement-title") or "").strip()
            track["composer"] = (root.findtext("./identification/creator") or "").strip()
            track["score_sha256"] = hashlib.sha256(score).hexdigest()
        tracks.append(track)
    result = {"record": RECORD, "archive_bytes": size, "archive_checksum": archive.get("checksum"),
                      "entry_count": len(entries),
                      "metadata_entries": metadata, "metadata_columns": list(rows[0]), "tracks": tracks,
                      "midi_count": sum(x[0].endswith(".mid") for x in entries),
                      "match_count": sum(x[0].endswith(".match") for x in entries),
                      "xml_count": sum(x[0].endswith(".musicxml") for x in entries)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "tracks": len(tracks),
                      "duration_ge_90s": [x["track"] for x in tracks if x["duration_sec"] >= 90],
                      "duration_range_sec": [min(x["duration_sec"] for x in tracks),
                                             max(x["duration_sec"] for x in tracks)]}, indent=2))


if __name__ == "__main__":
    main()
