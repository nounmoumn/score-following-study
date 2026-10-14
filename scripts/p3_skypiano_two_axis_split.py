#!/usr/bin/env python3
"""Deterministic participant-and-piece split for the SKY-Piano full release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SALT = "p3-skypiano-two-axis-v1"
PUBLIC_SAMPLE_IDS = {
    "240828_C0_N4_T2_Clementi_Op36_No1",
    "240828_C0_N5_T2_Bach_MinuetG",
    "240828_C0_N6_T2_Schubert_Op90_No4",
    "240828_C0_N7_T2_Chopin_Prelude_Op28_No4",
    "240828_C0_N8_T2_Mozart_K545",
}


def digest(kind, identifier):
    return hashlib.sha256(f"{SALT}:{kind}:{identifier}".encode()).hexdigest()


def alternating_assignment(items, identifier_key, stratum_key, kind):
    assignment = {}
    strata = sorted({str(item[stratum_key]) for item in items})
    for stratum in strata:
        identifiers = sorted(
            {str(item[identifier_key]) for item in items if str(item[stratum_key]) == stratum},
            key=lambda value: digest(kind, value),
        )
        for index, identifier in enumerate(identifiers):
            assignment[identifier] = "development" if index % 2 == 0 else "validation"
    return assignment


def build_split(trials):
    required = {"trial_id", "participant_id", "expertise", "piece_id", "piece_stratum", "duration_sec"}
    for trial in trials:
        missing = required - set(trial)
        if missing:
            raise ValueError(f"trial {trial.get('trial_id')} missing {sorted(missing)}")
    eligible = [trial for trial in trials if trial["trial_id"] not in PUBLIC_SAMPLE_IDS and float(trial["duration_sec"]) >= 45.0]
    participant_assignment = alternating_assignment(eligible, "participant_id", "expertise", "participant")
    piece_assignment = alternating_assignment(eligible, "piece_id", "piece_stratum", "piece")
    rows = []
    for trial in eligible:
        p = participant_assignment[str(trial["participant_id"])]
        w = piece_assignment[str(trial["piece_id"])]
        cell = "development" if p == w == "development" else "primary_lockbox" if p == w == "validation" else "cross_unused"
        rows.append({**trial, "participant_split": p, "piece_split": w, "cell": cell})
    lockbox = [row for row in rows if row["cell"] == "primary_lockbox"]
    counts = {
        "eligible_trials": len(rows),
        "participants": len({row["participant_id"] for row in rows}),
        "pieces": len({row["piece_id"] for row in rows}),
        "development_trials": sum(row["cell"] == "development" for row in rows),
        "primary_lockbox_trials": len(lockbox),
        "cross_unused_trials": sum(row["cell"] == "cross_unused" for row in rows),
        "lockbox_participants": len({row["participant_id"] for row in lockbox}),
        "lockbox_pieces": len({row["piece_id"] for row in lockbox}),
    }
    gates = {
        "participants_ge_10": counts["participants"] >= 10,
        "pieces_ge_10": counts["pieces"] >= 10,
        "eligible_trials_ge_30": counts["eligible_trials"] >= 30,
        "lockbox_participants_ge_5": counts["lockbox_participants"] >= 5,
        "lockbox_pieces_ge_5": counts["lockbox_pieces"] >= 5,
        "development_nonempty": counts["development_trials"] > 0,
        "sample_trials_excluded": not any(row["trial_id"] in PUBLIC_SAMPLE_IDS for row in rows),
    }
    return {
        "salt": SALT,
        "counts": counts,
        "pre_outcome_split_gate_passed": all(gates.values()),
        "gates": gates,
        "participant_assignment": participant_assignment,
        "piece_assignment": piece_assignment,
        "trials": rows,
        "temporal_outcomes_opened": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/p3_skypiano_two_axis_split/manifest.json")
    args = parser.parse_args()
    raw = json.loads(args.manifest.read_text(encoding="utf8"))
    trials = raw["trials"] if isinstance(raw, dict) else raw
    result = build_split(trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
