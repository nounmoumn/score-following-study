"""Fit and freeze the development relocation validator before lockbox use."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from matchmaker_ambiguity_rescue import fit_ridge  # noqa: E402


DEV = ROOT/"results"/"online_relocation_validation"
PROTOCOL = ROOT/"ONLINE_RELOCATION_LOCKBOX_PROTOCOL.md"
OUT = ROOT/"output"/"online_relocation_validator_frozen"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialise(model: dict[str, np.ndarray]) -> dict[str, object]:
    return {key: value.tolist() for key, value in model.items()}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    events_path = DEV/"events.json"
    result_path = DEV/"result.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    groups = np.array([event["work"] for event in events])
    target = np.array([event["target"] for event in events], float)
    immediate = np.array([event["immediate"] for event in events], float)
    delayed = np.array([event["delayed"] for event in events], float)
    work_targets = [target[groups == work].mean() for work in np.unique(groups)]
    global_action = "accept" if np.mean(work_targets) >= 0 else "hold"
    if global_action != "accept":
        raise ValueError("Development global action changed")
    sources = [Path(__file__), PROTOCOL, events_path, result_path, ROOT/"scripts"/"online_relocation_validation.py"]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in sources}
    model = {
        "version": 1,
        "delay_sec": 2.0,
        "jump_threshold_score_sec": 2.0,
        "ridge_alpha": 1.0,
        "prediction_threshold": 0.0,
        "global_action": global_action,
        "development_events": len(events),
        "development_works": len(np.unique(groups)),
        "immediate_feature_count": int(immediate.shape[1]),
        "delayed_feature_count": int(delayed.shape[1]),
        "immediate": serialise(fit_ridge(immediate, target, groups)),
        "delayed": serialise(fit_ridge(delayed, target, groups)),
        "source_hashes": freeze,
    }
    OUT.mkdir(parents=True)
    (OUT/"model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (OUT/"freeze.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "global_action": global_action, "events": len(events), "model_sha256": digest(OUT/"model.json")}, indent=2))


if __name__ == "__main__":
    main()
