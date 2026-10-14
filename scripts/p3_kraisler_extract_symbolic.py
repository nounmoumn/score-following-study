"""Range-extract only KRAISLER performance MIDI, piano scores, and matches."""

import csv
import hashlib
import io
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/api/records/21082251"
OUT = ROOT / "data/kraisler_symbolic"
MANIFEST = ROOT / "results/p3_external_metadata/kraisler_symbolic_manifest.json"


def get_range(url, start, end):
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                if response.status != 206 or len(data) != end - start + 1:
                    raise RuntimeError("range request mismatch")
                return data
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            if attempt == 4: raise
            time.sleep(2 ** attempt)


def main():
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    archive = next(f for f in record["files"] if f["key"] == "KRAISLER.zip")
    url, size = archive["links"]["self"], archive["size"]
    tail = get_range(url, max(0, size - 65557), size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack_from("<IHHHHIIH", tail, pos)
    central = get_range(url, cd_offset, cd_offset + cd_size - 1)
    stream = io.BytesIO(central); entries = {}
    for _ in range(count):
        header = stream.read(46); values = struct.unpack("<IHHHHHHIIIHHHHHII", header)
        compressed_size, uncompressed_size = values[8], values[9]
        name_len, extra_len, comment_len = values[10:13]
        name = stream.read(name_len).decode("utf-8", errors="replace")
        stream.seek(extra_len + comment_len, 1)
        entries[name] = (name, compressed_size, uncompressed_size, values[16], values[4])

    def extract(name):
        _, compressed_size, _, offset, method = entries[name]
        local = get_range(url, offset, offset + 29)
        values = struct.unpack("<IHHHHHIIIHH", local)
        payload_offset = offset + 30 + values[-2] + values[-1]
        payload = get_range(url, payload_offset, payload_offset + compressed_size - 1)
        return zlib.decompress(payload, -15) if method == 8 else payload

    metadata_name = next(name for name in entries if name.endswith("metadata.csv"))
    metadata = extract(metadata_name)
    rows = list(csv.DictReader(io.StringIO(metadata.decode("utf-8-sig"))))
    wanted = {metadata_name}
    for row in rows:
        for column in ("performance_midi_PF", "score_musicxml_PF", "alignment_match"):
            wanted.add("KRAISLER/" + row[column])
    output_rows = []
    for name in sorted(wanted):
        data = extract(name)
        relative = name.removeprefix("KRAISLER/")
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        output_rows.append({"archive_path": name, "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    result = {"record": RECORD, "archive_checksum": archive.get("checksum"),
              "selected_files": len(output_rows), "tracks": len(rows), "files": output_rows,
              "scope": "symbolic source acquisition only; no tracker or P3 outcome"}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
