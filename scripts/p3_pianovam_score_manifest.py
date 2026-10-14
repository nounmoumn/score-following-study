#!/usr/bin/env python3
"""Build an outcome-blind PianoVAM-to-score acquisition manifest."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PVM = ROOT / "results/p3_external_metadata/PianoVAM_metadata.json"
PC = ROOT / "results/p3_external_metadata/PianoCoRe_metadata.csv"
OUT = ROOT / "results/p3_pianovam_score_manifest.json"

# (PianoVAM composer regex, piece regex, PianoCoRe composer regex,
#  composition regex, optional movement regex). Rules encode identities, not similarity thresholds.
RULES = (
    (r"chopin", r"grande valse.*op\.?\s*18", r"chopin", r"(?:waltz.*op\.?\s*18|grande.*valse)", None),
    (r"beethoven", r"sonata no\.?\s*23 mvt\.?\s*1", r"beethoven", r"sonata no\.?\s*23.*op\.?\s*57", r"^1\D"),
    (r"beethoven", r"sonata no\.?\s*23 mvt\.?\s*3", r"beethoven", r"sonata no\.?\s*23.*op\.?\s*57", r"^3\D"),
    (r"debussy", r"clair de lune", r"debussy", r"suite bergamasque", r"clair de lune"),
    (r"mozart", r"sonata no\.?\s*16.*k\.?\s*545.*1st mvt", r"mozart", r"sonata no\.?\s*16.*k\.?\s*545", r"^1\D"),
    (r"chopin", r"nocturne op\.?\s*9 no\.?\s*3", r"chopin", r"nocturnes.*op\.?\s*9", r"no\.?\s*3\D"),
    (r"bach", r"italian concerto mvt\.?\s*1", r"bach", r"italian concerto.*971", None),
    (r"bach", r"italian concerto mvt\.?\s*3", r"bach", r"italian concerto.*971", None),
    (r"ravel", r"jeux d.?eau", r"ravel", r"jeux d.?eau.*(?:m\.?\s*30)?", None),
    (r"ravel", r"sonatine mvt\.?\s*1", r"ravel", r"sonatine.*m\.?\s*40", r"^1\D"),
    (r"rachmaninoff", r"prelude op\.?\s*32 no\.?\s*10", r"rachmaninoff", r"preludes.*op\.?\s*32", r"no\.?\s*10\D"),
    (r"debussy", r"images 3\.? mouvement", r"debussy", r"images.*(?:book 1|l\.?\s*110)", r"3.*mouvement"),
    (r"rachmaninoff", r"etude.?tableaux op\.?\s*39 no\.?\s*8", r"rachmaninoff", r"etudes.?tableaux.*op\.?\s*39", r"no\.?\s*8\D"),
    (r"chopin", r"ballade no\.?\s*1", r"chopin", r"ballade no\.?\s*1.*op\.?\s*23", None),
    (r"ravel", r"gaspard.*ondine", r"ravel", r"gaspard.*m\.?\s*55", r"ondine"),
    (r"beethoven", r"f[uü]r elise", r"beethoven", r"f[uü]r elise.*59", None),
    (r"rachmaninoff", r"moment.*op\.?\s*16 no\.?\s*4", r"rachmaninoff", r"moments musicaux.*op\.?\s*16", r"no\.?\s*4\D"),
    (r"satie", r"gymnopedie no\.?\s*1", r"satie", r"gymnop", r"^1\D"),
    (r"rachmaninoff", r"prelude op\.?\s*3 no\.?\s*2", r"rachmaninoff", r"(?:morceaux.*op\.?\s*3|prelude.*op\.?\s*3)", r"(?:no\.?\s*2|prelude)"),
)


def clean(value):
    return re.sub(r"_+", " ", value.lower()).replace(",", " ")


def matches(value, pattern):
    return pattern is None or re.search(pattern, clean(value), re.I) is not None


def truth(value):
    return str(value).lower() in {"true", "1", "yes"}


def main():
    pvm = list(json.loads(PVM.read_text(encoding="utf-8")).values())
    selected = []
    for row in pvm:
        if row["performance_method"] != "Solo":
            continue
        for index, rule in enumerate(RULES):
            if matches(row["composer"], rule[0]) and matches(row["piece"], rule[1]):
                selected.append((index, row)); break

    needed = set(index for index, _ in selected)
    candidates = defaultdict(list); offsets = defaultdict(int)
    with PC.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", ""); offset = offsets[split]; offsets[split] += 1
            for index in needed:
                rule = RULES[index]
                if (matches(row.get("composer", ""), rule[2]) and
                    matches(row.get("composition", ""), rule[3]) and
                    matches(row.get("movement", ""), rule[4]) and row.get("score_midi_path")):
                    quality = min(float(row.get("raw_precision") or 0), float(row.get("raw_recall") or 0))
                    priority = (row.get("score_dataset") != "ASAP", truth(row.get("tier_a_star")), quality,
                                row.get("score_id", ""))
                    candidates[index].append((priority, row, offset))

    score_choice = {}
    ambiguity = {}
    for index in needed:
        pool = candidates[index]
        works = {(row["composer"], row["composition"], row["movement"]) for _, row, _ in pool}
        ambiguity[index] = len(works)
        if pool:
            priority, row, offset = max(pool, key=lambda item: item[0])
            score_choice[index] = {"split": row["split"], "dataset_viewer_offset": offset,
                                   "composer": row["composer"], "composition": row["composition"],
                                   "movement": row["movement"], "score_dataset": row["score_dataset"],
                                   "score_id": row["score_id"], "raw_precision": row["raw_precision"],
                                   "raw_recall": row["raw_recall"], "candidate_work_count": len(works)}

    rows = []
    for index, row in selected:
        score = score_choice.get(index)
        rows.append({"record_time": row["record_time"], "split": row["split"],
                     "composer": row["composer"], "piece": row["piece"], "duration": row["duration"],
                     "performer": row["P1_name"], "rule_index": index,
                     "midi_repo_path": f"MIDI/{row['record_time']}.mid", "score": score,
                     "catalog_match": score is not None and score["candidate_work_count"] == 1})
    works = {row["rule_index"] for row in rows if row["catalog_match"]}
    composers = {row["score"]["composer"] for row in rows if row["catalog_match"]}
    gates = {"direct_midi_ge_20": len(rows) >= 20, "work_families_ge_10": len(works) >= 10,
             "composers_ge_8": len(composers) >= 8,
             "score_matched_works_ge_8": len(works) >= 8}
    result = {"scope": "Outcome-blind PianoVAM direct-MIDI to third-party-score acquisition manifest",
              "source_commit": "7aa9d7d8c061b7127cfd2fc6c3cd66bc441b94b8",
              "metadata_sha256": hashlib.sha256(PVM.read_bytes()).hexdigest(),
              "counts": {"recordings": len(rows), "matched_work_families": len(works),
                         "matched_composers": len(composers),
                         "splits": dict(Counter(row["split"] for row in rows))},
              "gates": gates, "continue_to_file_acquisition": all(gates.values()),
              "validation_eligible": False,
              "caveat": "Scores are catalog matches only; practice takes still require segmented alignment and independent review.",
              "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
