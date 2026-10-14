"""Range-extract the fixed 32-performance prospective Vienna subset."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vienna4x22_audio_mean_evidence"
MANIFEST = OUT / "manifest.json"
URL = "https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/audio.zip"
PERFORMERS = ("p02", "p05", "p08", "p11", "p14", "p17", "p20", "p22")
WORKS = (
    ("audio/Chopin_Ballade", "Chopin_op38"),
    ("audio/Chopin_Etude", "Chopin_op10_no3"),
    ("audio/Mozart", "Mozart_K331_1st-mov"),
    ("audio/Schubert", "Schubert_D783_no15"),
)
MEMBERS = tuple(f"{folder}/{stem}_{performer}.wav" for folder, stem in WORKS for performer in PERFORMERS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_group(group: tuple[str, ...], expected_sizes: dict[str, int]) -> list[dict[str, object]]:
    rows = []
    with RemoteZip(URL) as archive:
        for member in group:
            target = OUT / Path(member).name
            if target.exists() and target.stat().st_size == expected_sizes[member]:
                print(f"reuse {member}", flush=True)
            else:
                if target.exists():
                    resolved = target.resolve()
                    if resolved.parent != OUT.resolve() or resolved.name != Path(member).name:
                        raise RuntimeError(f"unsafe partial target: {resolved}")
                    print(
                        f"replace partial {member} ({target.stat().st_size}/{expected_sizes[member]} bytes)",
                        flush=True,
                    )
                    target.unlink()
                print(f"extract {member}", flush=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    while block := source.read(1024 * 1024):
                        sink.write(block)
                if target.stat().st_size != expected_sizes[member]:
                    raise RuntimeError(f"size mismatch: {target}")
            rows.append({
                "member": member,
                "file": str(target.relative_to(ROOT)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
    return rows


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"Refuse overwrite: {MANIFEST}")
    OUT.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as archive:
        infos = {info.filename: info.file_size for info in archive.infolist()}
        names = set(infos)
        missing = sorted(set(MEMBERS) - names)
        if missing:
            raise RuntimeError(f"missing fixed members: {missing}")
    groups = tuple(tuple(MEMBERS[index::4]) for index in range(4))
    with ThreadPoolExecutor(max_workers=4) as pool:
        grouped = list(pool.map(lambda group: extract_group(group, infos), groups))
    by_member = {row["member"]: row for rows in grouped for row in rows}
    rows = [by_member[member] for member in MEMBERS]
    value = {
        "source": URL,
        "selection": "all 4 works x fixed unseen performers p02,p05,p08,p11,p14,p17,p20,p22",
        "rows": rows,
    }
    MANIFEST.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
