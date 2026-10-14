"""Group SMD MIDI by work family and flag known ASAP work overlap (metadata only)."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/p3_external_metadata/smd_preflight.json"
OUTPUT = ROOT / "results/p3_external_metadata/smd_family_gate.json"

# These aliases are based on work catalogue numbers, not similar-looking titles.
USED_ASAP = {
    ("Bach", "BWV875"): "Bach Prelude/Fugue BWV875",
    ("Bach", "BWV888"): "Bach Fugue BWV888",
    ("Beethoven", "Op027No1"): "Beethoven Sonata 13, Op.27 No.1",
    ("Beethoven", "Op031No2"): "Beethoven Sonata 17, Op.31 No.2",
    ("Chopin", "Op010"): "Chopin Etudes Op.10 Nos.3/4",
}


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    by_family = defaultdict(list)
    for row in data["rows"]:
        by_family[(row["composer"].replace("Rachmaninov", "Rachmaninoff"), row["family_code"])].append(row)
    families = []
    for (composer, code), rows in sorted(by_family.items()):
        long_rows = [row for row in rows if row["duration_sec"] >= 90]
        overlap = USED_ASAP.get((composer, code))
        families.append({"composer": composer, "family_code": code,
                         "performance_count": len(rows), "ge_90s": len(long_rows),
                         "max_duration_sec": max(row["duration_sec"] for row in rows),
                         "known_asap_overlap": overlap,
                         "status": "known_overlap" if overlap else "needs_independent_score_and_gt",
                         "files": [row["filename"] for row in rows]})
    output = {"source": data["source"], "families": families,
              "family_count": len(families),
              "long_family_count": sum(f["ge_90s"] > 0 for f in families),
              "long_without_known_asap_overlap": sum(f["ge_90s"] > 0 and not f["known_asap_overlap"] for f in families),
              "candidate_composer_count": len({f["composer"] for f in families
                                                if f["ge_90s"] > 0 and not f["known_asap_overlap"]}),
              "known_overlap_families": [f["composer"] + "/" + f["family_code"] for f in families
                                          if f["known_asap_overlap"]],
              "caveat": "Known catalogue overlap only. Remaining families are score/GT construction candidates, not eligible independent validation items."}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "families"}, indent=2))


if __name__ == "__main__":
    main()
