"""Independent audit of the prospective Vienna mean-evidence validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "online_relocation_audio_mean_evidence"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
EXPECTED_MODEL = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"
METHODS = ("arzt", "outerhmm")
PERFORMERS = ("p02", "p05", "p08", "p11", "p14", "p17", "p20", "p22")
STEMS = ("Chopin_op38", "Chopin_op10_no3", "Mozart_K331_1st-mov", "Schubert_D783_no15")
CASES = tuple(f"{stem}_{performer}" for stem in STEMS for performer in PERFORMERS)
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(features: list[float], model: dict[str, list[float]]) -> float:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    if x.shape != mean.shape or scale.shape != mean.shape or len(coefficient) != len(x) + 1:
        raise AssertionError("feature/model shape mismatch")
    return float(coefficient[0] + ((x - mean) / scale) @ coefficient[1:])


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def close(left: object, right: object, tolerance: float = 1e-12) -> float:
    residual = abs(float(left) - float(right))
    if residual > tolerance:
        raise AssertionError(f"numeric mismatch: {left!r} != {right!r}")
    return residual


def main() -> None:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    events = json.loads((OUT / "events.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    freeze_drift = [name for name, expected in freeze.items() if digest(ROOT / name) != expected]
    source_drift = [
        name for name, expected in model.get("source_hashes", {}).items()
        if digest(ROOT / name) != expected
    ]
    if digest(MODEL_PATH) != EXPECTED_MODEL or freeze_drift or source_drift:
        raise AssertionError("frozen input drift")

    concatenated = []
    implementation = True
    case_counts = {}
    for case_id in CASES:
        case = json.loads((OUT / f"{case_id}.json").read_text(encoding="utf-8"))
        query = np.asarray(case["query_time"], dtype=float)
        truth = np.asarray(case["truth"], dtype=float)
        implementation = implementation and bool(
            len(query) >= 20 and len(query) == len(truth)
            and np.isfinite(query).all() and np.isfinite(truth).all()
            and np.all(np.diff(query) > 0) and np.all(np.diff(truth) > 0)
            and case["gt_audit"]["dropped_nonmonotone_or_duplicate"] == 0
        )
        for method in METHODS:
            tracker = case["trackers"][method]
            path = np.asarray(tracker["path_seconds"], dtype=float)
            implementation = implementation and bool(
                path.ndim == 2 and path.shape[0] == 2 and path.shape[1] >= 20
                and np.isfinite(path).all() and np.all(np.diff(path[0]) >= -1e-12)
                and np.all(path[1] >= -1e-7)
                and np.all(path[1] <= float(case["score_duration"]) + 1e-7)
                and tracker["path_valid"]
            )
        case_counts[case_id] = len(case["events"])
        concatenated.extend(case["events"])
    if concatenated != events or len({event["event_id"] for event in events}) != len(events):
        raise AssertionError("case aggregation mismatch")

    maximum_residual = 0.0
    rebuilt = []
    for event in events:
        immediate = predict(event["immediate"], model["immediate"])
        delayed = predict(event["delayed"], model["delayed"])
        hold_loss = float(event["hold_loss"])
        accept_loss = float(event["accept_loss"])
        oracle = min(hold_loss, accept_loss)
        action = "accept" if float(event["delayed"][15]) >= float(event["delayed"][14]) else "hold"
        mean_regret = (accept_loss if action == "accept" else hold_loss) - oracle
        checks = {
            "immediate_prediction": immediate,
            "delayed_prediction": delayed,
            "global_regret": accept_loss - oracle,
            "immediate_regret": (accept_loss if immediate >= 0 else hold_loss) - oracle,
            "delayed_regret": (accept_loss if delayed >= 0 else hold_loss) - oracle,
            "mean_evidence_margin": float(event["delayed"][15]) - float(event["delayed"][14]),
            "mean_evidence_regret": mean_regret,
        }
        for key, value in checks.items():
            maximum_residual = max(maximum_residual, close(value, event[key]))
        if event["mean_evidence_action"] != action:
            raise AssertionError("mean-evidence action mismatch")
        rebuilt.append({**event, **checks, "mean_evidence_action": action})

    works = sorted({event["work"] for event in rebuilt})
    performances = sorted({event["case"] for event in rebuilt})
    methods = sorted({event["method"] for event in rebuilt})
    per_work = []
    for work in works:
        subset = [event for event in rebuilt if event["work"] == work]
        per_work.append({
            "work": work, "events": len(subset),
            "performances": len({event["case"] for event in subset}),
            "global_regret": float(np.mean([event["global_regret"] for event in subset])),
            "frozen_delayed_regret": float(np.mean([event["delayed_regret"] for event in subset])),
            "mean_evidence_regret": float(np.mean([event["mean_evidence_regret"] for event in subset])),
        })
    if per_work != result["per_work"]:
        raise AssertionError("per-work mismatch")
    difference = np.asarray(
        [row["global_regret"] - row["mean_evidence_regret"] for row in per_work], dtype=float
    )
    ci = bootstrap_ci(difference) if len(difference) else [None, None]
    holds = [event for event in rebuilt if event["mean_evidence_action"] == "hold"]
    precision = float(np.mean([event["oracle_action"] == "hold" for event in holds])) if holds else None
    gates = {
        "implementation": bool(implementation),
        "support": len(rebuilt) >= 20 and len(performances) >= 12 and len(works) >= 3
        and methods == list(METHODS),
        "work_equal_improvement": bool(len(difference) and difference.mean() > 0),
        "bootstrap_lower_positive": bool(len(difference) and ci[0] > 0),
        "no_material_work_worsening": bool(
            len(difference) and np.all(-difference <= 0.05 + 1e-12)
        ),
        "hold_precision": bool(len(holds) >= 5 and precision is not None and precision >= 0.8),
    }
    gates["pass"] = all(gates.values())
    if gates != result["gates"]:
        raise AssertionError("gate mismatch")
    maximum_residual = max(maximum_residual, close(difference.mean(), result["work_equal_improvement"]))
    for left, right in zip(ci, result["bootstrap95"]):
        maximum_residual = max(maximum_residual, close(left, right))

    audit = {
        "status": "PASS",
        "validation_status_reproduced": "PASS" if gates["pass"] else "FAIL",
        "events": len(rebuilt), "works": len(works), "performances": len(performances),
        "methods": methods, "case_event_counts": case_counts,
        "mean_evidence_holds": len(holds), "hold_precision": precision,
        "maximum_numeric_residual": maximum_residual,
        "freeze_hash_drift": freeze_drift, "development_source_hash_drift": source_drift,
        "gates": gates,
    }
    (OUT / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
