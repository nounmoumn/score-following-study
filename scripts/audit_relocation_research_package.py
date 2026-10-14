"""Independent top-level audit of the frozen relocation research package.

This script does not import any experiment implementation.  It checks the
persisted experiment results, their independent audits, frozen-model hash, and
the exact success/failure boundary used in the paper draft.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "relocation_research_package_audit.json"
MODEL_SHA256 = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"


def read_json(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tol: float = 1e-12) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def main() -> None:
    dev = read_json("results/online_relocation_validation/result.json")
    dev_audit = read_json("results/online_relocation_validation/independent_audit.json")
    lock = read_json("results/online_relocation_validation_lockbox/result.json")
    lock_audit = read_json(
        "results/online_relocation_validation_lockbox/independent_audit.json"
    )
    audio = read_json("results/online_relocation_vienna_audio/result.json")
    audio_audit = read_json(
        "results/online_relocation_vienna_audio/independent_audit.json"
    )
    rescue = read_json("results/online_relocation_audio_mean_evidence/result.json")
    rescue_audit = read_json(
        "results/online_relocation_audio_mean_evidence/independent_audit.json"
    )

    model_path = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    checks = {
        "frozen_model_hash": model_hash == MODEL_SHA256,
        "development_gate_failed_only_positive_work_count": (
            dev["gates"]["pass"] is False
            and dev["gates"]["support"] is True
            and dev["gates"]["beats_global_ci"] is True
            and dev["gates"]["positive_works"] is False
        ),
        "development_audit_reproduced": (
            dev_audit["status"] == "PASS"
            and dev_audit["gates_recomputed"] == dev["gates"]
            and close(dev_audit["max_fold_residual"], 0.0)
        ),
        "midi_lockbox_pass": lock["status"] == "PASS" and lock["gates"]["pass"],
        "midi_lockbox_effect": (
            lock["eligible_events"] == 61
            and close(lock["work_equal"]["global_regret"], 0.7002971969336257)
            and close(lock["work_equal"]["delayed_regret"], 0.3343583691418718)
            and lock["work_equal"]["global_minus_delayed_ci"][0] > 0
        ),
        "midi_lockbox_independent_audit": (
            lock_audit["status"] == "PASS"
            and lock_audit["lockbox_status_reproduced"] == "PASS"
            and close(lock_audit["max_numeric_residual"], 0.0)
            and not lock_audit["lockbox_input_hash_drift"]
            and not lock_audit["development_source_hash_drift"]
        ),
        "audio_transfer_failed": (
            audio["status"] == "FAIL"
            and audio["events"] == 5
            and audio["methods"] == ["outerhmm"]
            and audio["holds"] == 0
        ),
        "audio_transfer_audited": (
            audio_audit["status"] == "PASS"
            and audio_audit["frozen_transfer_status_reproduced"] == "FAIL"
            and close(audio_audit["max_numeric_residual"], 0.0)
        ),
        "audio_rescue_failed": (
            rescue["status"] == "FAIL"
            and rescue["events"] == 9
            and rescue["methods"] == ["outerhmm"]
            and close(rescue["hold_precision"], 0.6)
            and rescue["work_equal_improvement"] < 0
            and rescue["bootstrap95"][1] <= 0
        ),
        "audio_rescue_independent_audit": (
            rescue_audit["status"] == "PASS"
            and rescue_audit["validation_status_reproduced"] == "FAIL"
            and close(rescue_audit["maximum_numeric_residual"], 0.0)
            and not rescue_audit["freeze_hash_drift"]
            and not rescue_audit["development_source_hash_drift"]
        ),
        "audio_trigger_coverage_boundary": (
            audio["works"] == 2
            and rescue["works"] == 2
            and audio["methods"] == rescue["methods"] == ["outerhmm"]
        ),
    }

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "purpose": "reproduce the paper-level success/failure boundary without importing experiment code",
        "checks": checks,
        "frozen_model_sha256": model_hash,
        "paper_boundary": {
            "development": "FAIL (predeclared positive-work count only)",
            "midi_lockbox": "PASS",
            "frozen_audio_transfer": "FAIL",
            "prospective_audio_rescue": "FAIL",
            "permitted_conclusion": (
                "event-conditioned validation reduced regret on unseen symbolic works, "
                "but neither trigger coverage nor the decision rule transferred to audio"
            ),
        },
        "key_numbers": {
            "midi_lockbox_events": lock["eligible_events"],
            "midi_lockbox_global_regret": lock["work_equal"]["global_regret"],
            "midi_lockbox_delayed_regret": lock["work_equal"]["delayed_regret"],
            "midi_lockbox_improvement_ci": lock["work_equal"][
                "global_minus_delayed_ci"
            ],
            "audio_initial_events": audio["events"],
            "audio_prospective_events": rescue["events"],
            "audio_rescue_improvement": rescue["work_equal_improvement"],
            "audio_rescue_ci": rescue["bootstrap95"],
            "audio_rescue_hold_precision": rescue["hold_precision"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
