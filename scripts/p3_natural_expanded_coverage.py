#!/usr/bin/env python3
"""Apply frozen event/coverage methods to the fixed expanded cohort."""

import json
from pathlib import Path

try:
    from scripts import p3_natural_coverage_units as coverage
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_coverage_units as coverage
    import p3_natural_event_restart_candidates as event

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "results/p3_natural_smd_exact_supplement.json"
OUT = ROOT / "results/p3_natural_expanded_coverage.json"


def main():
    supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
    records = event.records() + [row for row in supplement["rows"] if row["eligible"]]
    rows = []
    for record in records:
        candidate_row = event.process(record)
        unit_row = coverage.process(record, candidate_row)
        unit_row["content_mapping_weak"] = candidate_row["content_mapping_weak"]
        unit_row["unique_anchor_coverage"] = candidate_row["unique_anchor_coverage"]
        rows.append(unit_row)
    ready = [row for row in rows if row["reviewable_units"]]
    gates = {"all_fixed_pairs_included": len(records) == 29,
             "recordings_with_reviewable_unit_ge_20": len(ready) >= 20,
             "represented_composers_ge_8": len({r["composer"] for r in ready}) >= 8,
             "all_units_have_gate_provenance": all("gates" in u for r in rows for u in r["units"])}
    result = {"scope": "Expanded outcome-blind coverage-unit audit; no tracker/P3 outcomes",
        "method_parameters": {"event": {"ngram_events": event.NGRAM_EVENTS,
            "performance_onset_tolerance_sec": event.PERFORMANCE_ONSET_TOLERANCE,
            "max_performance_event_gap": event.MAX_PERFORMANCE_EVENT_GAP},
            "coverage": {"minimum_duration_sec": coverage.MIN_DURATION_SEC,
            "minimum_anchors": coverage.MIN_ANCHORS, "minimum_score_span": coverage.MIN_SCORE_SPAN,
            "minimum_monotone_share": coverage.MIN_MONOTONE_SHARE}},
        "counts": {"recordings": len(rows), "recordings_with_reviewable_unit": len(ready),
            "reviewable_units": sum(r["reviewable_units"] for r in rows),
            "represented_composers": len({r["composer"] for r in ready}),
            "content_mapping_weak": sum(r["content_mapping_weak"] for r in rows)},
        "gates": gates, "continue_to_boundary_review": all(gates.values()),
        "validation_eligible": False, "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
