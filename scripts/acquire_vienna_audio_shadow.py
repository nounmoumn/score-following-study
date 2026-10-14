"""Range-extract the outcome-blind Vienna audio-shadow subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from remotezip import RemoteZip


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vienna4x22_audio_shadow"
MANIFEST = OUT / "manifest.json"
URL = "https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/data/audio.zip"
MEMBERS = (
    "audio/Chopin_Ballade/Chopin_op38_p01.wav",
    "audio/Chopin_Ballade/Chopin_op38_p12.wav",
    "audio/Chopin_Etude/Chopin_op10_no3_p01.wav",
    "audio/Chopin_Etude/Chopin_op10_no3_p12.wav",
    "audio/Mozart/Mozart_K331_1st-mov_p01.wav",
    "audio/Mozart/Mozart_K331_1st-mov_p12.wav",
    "audio/Schubert/Schubert_D783_no15_p01.wav",
    "audio/Schubert/Schubert_D783_no15_p12.wav",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"Refuse overwrite: {MANIFEST}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with RemoteZip(URL) as archive:
        names = set(archive.namelist())
        missing = sorted(set(MEMBERS) - names)
        if missing:
            raise RuntimeError(f"Missing fixed members: {missing}")
        for member in MEMBERS:
            target = OUT / Path(member).name
            if target.exists():
                raise RuntimeError(f"Refuse overwrite: {target}")
            print(f"extract {member}", flush=True)
            with archive.open(member) as source, target.open("wb") as sink:
                while block := source.read(1024 * 1024):
                    sink.write(block)
            rows.append({
                "member": member,
                "file": str(target.relative_to(ROOT)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
    value = {"source": URL, "selection": "all 4 works x fixed performers p01,p12", "rows": rows}
    MANIFEST.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
