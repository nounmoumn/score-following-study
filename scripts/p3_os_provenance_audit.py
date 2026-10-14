#!/usr/bin/env python3
"""Audit whether any P3-OS repaired-clock outcome was still unopened."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_os_provenance_audit"


def main():
    manifest = json.loads((ROOT / "results/p3_clock_repair_preflight/manifest.json").read_text())
    eligible = {(row["source"], row["unit"]) for row in manifest["rows"] if row["eligible"]}
    batik_rows = json.loads((ROOT / "results/p3_fhvb6_final_validation/temporal_rows.json").read_text())
    kraisler_rows = json.loads((ROOT / "results/p3_fhvb_long_horizon_validation/temporal_rows.json").read_text())
    opened = {
        *(('Batik', str(row['track'])) for row in batik_rows),
        *(('KRAISLER', str(row['track'])) for row in kraisler_rows),
    }
    overlap = eligible & opened
    result = {
        "eligible_repaired_tracks": len(eligible),
        "previously_opened_tracks": len(opened),
        "overlap_count": len(overlap),
        "overlap": [{"source": source, "unit": unit} for source, unit in sorted(overlap)],
        "unopened_at_p3_os_run": [
            {"source": source, "unit": unit} for source, unit in sorted(eligible - opened)
        ],
        "all_p3_os_outcomes_previously_opened": eligible <= opened,
        "p3_os_independent_validation": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
