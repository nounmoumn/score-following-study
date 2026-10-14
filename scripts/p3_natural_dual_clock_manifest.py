#!/usr/bin/env python3
"""Seal the disjoint one-shot ASAP validation set for the dual-clock selector."""

from __future__ import annotations

import json

try:
    from scripts import p3_natural_path_clock_manifest as source
except ImportError:
    import p3_natural_path_clock_manifest as source

ROOT = source.ROOT
PRIOR = ROOT / "results/p3_natural_path_clock_manifest.json"
OUT = ROOT / "results/p3_natural_dual_clock_manifest.json"


def main():
    prior = json.loads(PRIOR.read_text(encoding="utf8"))
    excluded = {row["work"] for row in prior["rows"]}
    eligible = [row for row in source.eligible_rows() if row["work"] not in excluded]
    selected = source.select(eligible)
    overlap = excluded & {row["work"] for row in selected}
    if len(selected) != source.TARGET or len({row["composer"] for row in selected}) < 6 or overlap:
        raise RuntimeError("Independent dual-clock manifest gate failed")
    for row in selected:
        row["sha256"] = {
            key: source.sha256(ROOT / row[key])
            for key in ("score_midi", "score_annotations", "performance_midi", "performance_annotations")
        }
    result = {
        "scope": "One-shot dual-clock selector validation; no tracker/P3 outcomes",
        "selection_algorithm": "prior 24 works excluded; path-hash performance; composer round-robin work-hash",
        "remaining_eligible_works": len(eligible),
        "selected_works": len(selected),
        "selected_composers": len({row["composer"] for row in selected}),
        "overlap_with_development": len(overlap),
        "rows": selected,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
