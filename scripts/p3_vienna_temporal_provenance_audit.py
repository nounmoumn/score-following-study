#!/usr/bin/env python3
"""Freeze the pre-P3-OS temporal provenance of Vienna 4x22."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_vienna_temporal_provenance_audit"
PREEXISTING = (
    "p3_fh_horizon_development.py",
    "p3_fhl_long_window_development.py",
    "p3_fu_blend_development.py",
    "p3_fu_vienna_development.py",
    "p3_vienna_forecast_preflight.py",
    "p3_vienna_forecast_validation.py",
    "p3_vienna_point_interval_diagnostic.py",
    "p3_vienna_preflight.py",
)


def main():
    manifest = json.loads((ROOT / "results/p3_vienna_forecast_preflight/manifest.json").read_text())
    validation = [row for row in manifest["rows"] if row["split"] == "validation"]
    inspections = []
    for name in PREEXISTING:
        text = (ROOT / "scripts" / name).read_text(encoding="utf8")
        inspections.append({
            "script": name,
            "mentions_sustained_start": "sustained_start" in text,
            "mentions_event_or_onset_output": any(token in text for token in ('"event"', '"onset_sec"')),
        })
    result = {
        "audit_frozen_before_vienna_p3_os_execution": True,
        "validation_performances": len(validation),
        "validation_works": sorted({row["work"] for row in validation}),
        "prior_vienna_scripts": inspections,
        "no_prior_natural_temporal_endpoint_code": not any(
            row["mentions_sustained_start"] or row["mentions_event_or_onset_output"] for row in inspections
        ),
        "position_and_help_sign_endpoints_previously_opened": True,
        "natural_drift_onset_and_cause_endpoints_previously_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
