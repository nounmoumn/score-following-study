"""Independent, read-only audit of the frozen relocation lockbox result."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "online_relocation_validation_lockbox"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
MANIFEST = ROOT / "output" / "ambiguity_map_fresh_cohort" / "manifest.csv"
SEED = 20260924
DERIVED_FIELDS = {
    "immediate_prediction", "delayed_prediction", "global_regret",
    "immediate_regret", "delayed_regret", "global_correct",
    "immediate_correct", "delayed_correct",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def predict(features: list[float], model: dict[str, list[float]]) -> float:
    """Reimplement the frozen affine ridge predictor without importing run code."""
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coefficient"], dtype=float)
    if x.shape != mean.shape or mean.shape != scale.shape or len(coef) != len(x) + 1:
        raise AssertionError("Frozen feature/model shape mismatch")
    return float(coef[0] + ((x - mean) / scale) @ coef[1:])


def close(a: object, b: object, tol: float = 1e-12) -> float:
    residual = abs(float(a) - float(b))
    if residual > tol:
        raise AssertionError(f"numeric mismatch: {a!r} != {b!r} (residual {residual})")
    return residual


def main() -> None:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    events = json.loads((OUT / "events.json").read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))

    drift = [relative for relative, expected in freeze.items() if digest(ROOT / relative) != expected]
    source_drift = [
        relative for relative, expected in model["source_hashes"].items()
        if digest(ROOT / relative) != expected
    ]
    if drift or source_drift:
        raise AssertionError(f"input drift: lockbox={drift}, development={source_drift}")
    if result["model_sha256"] != digest(MODEL_PATH):
        raise AssertionError("model digest mismatch")

    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    development = [row for row in rows if row["role"] == "development"]
    lockbox = [row for row in rows if row["role"] == "lockbox"]
    work_overlap = sorted({r["work_family"] for r in development} & {r["work_family"] for r in lockbox})
    performer_overlap = sorted({r["performer"] for r in development} & {r["performer"] for r in lockbox})
    if work_overlap or performer_overlap:
        raise AssertionError("development/lockbox identity overlap")

    case_events: list[dict[str, object]] = []
    case_counts: dict[str, int] = {}
    for row in lockbox:
        values = json.loads((OUT / f"{row['case_id']}_events.json").read_text(encoding="utf-8"))
        case_counts[row["case_id"]] = len(values)
        case_events.extend(values)
    stripped = [{k: v for k, v in event.items() if k not in DERIVED_FIELDS} for event in events]
    if case_events != stripped:
        raise AssertionError("concatenated per-case events differ from aggregate raw events")
    if len({event["event_id"] for event in events}) != len(events):
        raise AssertionError("duplicate event ids")

    recomputed: list[dict[str, object]] = []
    max_residual = 0.0
    for event in events:
        immediate = predict(event["immediate"], model["immediate"])
        delayed = predict(event["delayed"], model["delayed"])
        hold = float(event["hold_loss"])
        accept = float(event["accept_loss"])
        oracle = min(hold, accept)
        values = {
            "immediate_prediction": immediate,
            "delayed_prediction": delayed,
            "global_regret": accept - oracle,
            "immediate_regret": (accept if immediate >= 0 else hold) - oracle,
            "delayed_regret": (accept if delayed >= 0 else hold) - oracle,
            "global_correct": accept <= hold,
            "immediate_correct": (immediate >= 0) == (accept <= hold),
            "delayed_correct": (delayed >= 0) == (accept <= hold),
        }
        for key, value in values.items():
            if isinstance(value, bool):
                if bool(event[key]) != value:
                    raise AssertionError(f"boolean mismatch for {event['event_id']} {key}")
            else:
                max_residual = max(max_residual, close(event[key], value))
        if event["method"] not in {"arzt", "outerhmm"}:
            raise AssertionError("unexpected tracker")
        if event["oracle_action"] != ("accept" if accept <= hold else "hold"):
            raise AssertionError("oracle action mismatch")
        recomputed.append({**event, **values})

    works = sorted({str(event["work"]) for event in recomputed})
    per_work = []
    for work in works:
        subset = [event for event in recomputed if event["work"] == work]
        per_work.append({
            "work": work,
            "events": len(subset),
            **{
                key: float(np.mean([float(event[key]) for event in subset]))
                for key in (
                    "global_regret", "immediate_regret", "delayed_regret",
                    "global_correct", "immediate_correct", "delayed_correct",
                )
            },
        })

    g = np.asarray([row["global_regret"] for row in per_work])
    i = np.asarray([row["immediate_regret"] for row in per_work])
    d = np.asarray([row["delayed_regret"] for row in per_work])
    dg, di = g - d, i - d
    actionable = g > 1e-12
    floor = ~actionable
    required = int(np.ceil(0.6 * int(actionable.sum()))) if actionable.any() else 0
    gates = {
        "support": len(events) >= 40 and len(works) >= 6
        and sorted({event["method"] for event in events}) == ["arzt", "outerhmm"],
        "global_improvement_ci": float(dg.mean()) > 0 and bootstrap_ci(dg)[0] > 0,
        "immediate_improvement_ci": float(di.mean()) > 0 and bootstrap_ci(di)[0] > 0,
        "actionable_works": int(actionable.sum()) >= 3 and int(np.sum(dg[actionable] > 0)) >= required,
        "floor_noninferiority": bool(np.all(d[floor] <= 0.05 + 1e-12)),
        "no_materially_worsened_work": int(np.sum(d - g > 0.05)) == 0,
    }
    gates["pass"] = all(gates.values())
    if gates != result["gates"]:
        raise AssertionError(f"gate mismatch: {gates} != {result['gates']}")

    saved_by_work = {row["work"]: row for row in result["per_work"]}
    for row in per_work:
        saved = saved_by_work[row["work"]]
        if row["events"] != saved["events"]:
            raise AssertionError("per-work event count mismatch")
        for key in (
            "global_regret", "immediate_regret", "delayed_regret",
            "global_correct", "immediate_correct", "delayed_correct",
        ):
            saved_key = key.replace("correct", "accuracy")
            max_residual = max(max_residual, close(row[key], saved[saved_key]))

    summary_expected = {
        "global_regret": float(g.mean()),
        "immediate_regret": float(i.mean()),
        "delayed_regret": float(d.mean()),
        "global_minus_delayed": float(dg.mean()),
        "immediate_minus_delayed": float(di.mean()),
        "global_accuracy": float(np.mean([row["global_correct"] for row in per_work])),
        "immediate_accuracy": float(np.mean([row["immediate_correct"] for row in per_work])),
        "delayed_accuracy": float(np.mean([row["delayed_correct"] for row in per_work])),
    }
    for key, value in summary_expected.items():
        max_residual = max(max_residual, close(value, result["work_equal"][key]))
    for key, values in {
        "global_minus_delayed_ci": bootstrap_ci(dg),
        "immediate_minus_delayed_ci": bootstrap_ci(di),
    }.items():
        for value, saved in zip(values, result["work_equal"][key]):
            max_residual = max(max_residual, close(value, saved))

    nonzero = dg[dg > 1e-12]
    signed_means = np.asarray([
        np.mean(nonzero * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
    ])
    event_global = np.asarray([event["global_regret"] for event in recomputed], dtype=float)
    event_delayed = np.asarray([event["delayed_regret"] for event in recomputed], dtype=float)
    robustness = {
        "work_improvements": dg.tolist(),
        "leave_one_work_out_mean_improvement": [
            float(np.mean(np.delete(dg, index))) for index in range(len(dg))
        ],
        "minimum_leave_one_work_out_mean_improvement": float(min(
            np.mean(np.delete(dg, index)) for index in range(len(dg))
        )),
        "event_weighted_global_regret": float(event_global.mean()),
        "event_weighted_delayed_regret": float(event_delayed.mean()),
        "event_weighted_improvement": float((event_global - event_delayed).mean()),
        "exact_sign_flip_p_one_sided_nonzero_works": float(
            np.mean(signed_means >= float(nonzero.mean()) - 1e-15)
        ),
        "immediate_hold_actions": int(sum(event["immediate_prediction"] < 0 for event in recomputed)),
        "delayed_hold_actions": int(sum(event["delayed_prediction"] < 0 for event in recomputed)),
        "oracle_hold_events": int(sum(event["hold_loss"] < event["accept_loss"] for event in recomputed)),
        "delayed_hold_true_positives": int(sum(
            event["delayed_prediction"] < 0 and event["hold_loss"] < event["accept_loss"]
            for event in recomputed
        )),
    }

    audit = {
        "status": "PASS",
        "lockbox_status_reproduced": "PASS" if gates["pass"] else "FAIL",
        "events": len(events),
        "works_with_events": len(works),
        "case_event_counts": case_counts,
        "feature_counts": {
            "immediate": len(model["immediate"]["mean"]),
            "delayed": len(model["delayed"]["mean"]),
        },
        "development_lockbox_work_overlap": work_overlap,
        "development_lockbox_performer_overlap": performer_overlap,
        "max_numeric_residual": max_residual,
        "lockbox_input_hash_drift": drift,
        "development_source_hash_drift": source_drift,
        "gates": gates,
        "posthoc_robustness_diagnostics": robustness,
    }
    (OUT / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
