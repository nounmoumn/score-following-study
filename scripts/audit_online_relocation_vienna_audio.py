"""Independent, read-only audit of the frozen Vienna audio transfer run."""

from __future__ import annotations

import hashlib
import itertools
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "online_relocation_vienna_audio"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
EXPECTED_MODEL = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"
METHODS = ("arzt", "outerhmm")
DELAY = 2.0
HOP = 0.1
WINDOW = 20
SEED = 20260924
CASES = (
    "Chopin_op38_p01", "Chopin_op38_p12",
    "Chopin_op10_no3_p01", "Chopin_op10_no3_p12",
    "Mozart_K331_1st-mov_p01", "Mozart_K331_1st-mov_p12",
    "Schubert_D783_no15_p01", "Schubert_D783_no15_p12",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(features: list[float], model: dict[str, list[float]]) -> float:
    """Reimplement the affine ridge model without importing the run code."""
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    if x.shape != mean.shape or scale.shape != mean.shape or len(coefficient) != len(x) + 1:
        raise AssertionError("feature/model shape mismatch")
    return float(coefficient[0] + ((x - mean) / scale) @ coefficient[1:])


def burst_events(path: np.ndarray) -> list[int]:
    """Independent copy of the frozen >=2 score-second burst definition."""
    change = np.abs(np.diff(path[1]))
    candidates = [
        index for index in range(1, path.shape[1])
        if path[0, index] >= 10.0 and change[index - 1] >= 2.0
    ]
    if not candidates:
        return []
    bursts = [[candidates[0]]]
    for index in candidates[1:]:
        if path[0, index] - path[0, bursts[-1][-1]] <= 2.0:
            bursts[-1].append(index)
        else:
            bursts.append([index])
    return [
        max(burst, key=lambda index: abs(path[1, index] - path[1, index - 1]))
        for burst in bursts
    ]


def latest(path: np.ndarray, query: float) -> float | None:
    indexes = np.flatnonzero(path[0] <= query + 1e-12)
    return None if not len(indexes) else float(path[1, indexes[-1]])


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return float(handle.getnframes() / handle.getframerate())


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def close(left: object, right: object, tolerance: float = 1e-12) -> float:
    residual = abs(float(left) - float(right))
    if residual > tolerance:
        raise AssertionError(f"numeric mismatch {left!r} != {right!r}; residual={residual}")
    return residual


def main() -> None:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    aggregate = json.loads((OUT / "events.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    freeze_drift = [name for name, expected in freeze.items() if digest(ROOT / name) != expected]
    source_drift = [
        name for name, expected in model.get("source_hashes", {}).items()
        if digest(ROOT / name) != expected
    ]
    if digest(MODEL_PATH) != EXPECTED_MODEL or freeze_drift or source_drift:
        raise AssertionError(
            f"frozen input drift: model={digest(MODEL_PATH)}, freeze={freeze_drift}, source={source_drift}"
        )

    case_events: list[dict[str, object]] = []
    case_counts: dict[str, int] = {}
    funnel_by_method: dict[str, Counter[str]] = defaultdict(Counter)
    funnel_by_case: dict[str, dict[str, int]] = {}
    total_audio_seconds = 0.0
    tracker_runtime = Counter()
    implementation_checks = []

    for case_id in CASES:
        case = json.loads((OUT / f"{case_id}.json").read_text(encoding="utf-8"))
        if case["case"] != case_id:
            raise AssertionError("case identity mismatch")
        query = np.asarray(case["query_time"], dtype=float)
        truth = np.asarray(case["truth"], dtype=float)
        duration = wav_duration(Path(case["audio"]))
        total_audio_seconds += duration
        gt_ok = bool(
            len(query) >= 20 and len(query) == len(truth)
            and np.isfinite(query).all() and np.isfinite(truth).all()
            and np.all(np.diff(query) > 0) and np.all(np.diff(truth) > 0)
            and case["gt_audit"]["strict_monotone_anchors"] == len(query)
            and case["gt_audit"]["dropped_nonmonotone_or_duplicate"] == 0
        )
        if not gt_ok:
            raise AssertionError(f"GT audit failed: {case_id}")

        expected_events = list(case["events"])
        case_counts[case_id] = len(expected_events)
        case_events.extend(expected_events)
        expected_keys = {
            (event["method"], round(float(event["t0"]), 10),
             round(float(event["origin"]), 10), round(float(event["destination"]), 10))
            for event in expected_events
        }
        eligible_keys = set()
        local_funnel = Counter()
        score_frames = len(case["score_chroma"])

        for method in METHODS:
            tracker = case["trackers"][method]
            path = np.asarray(tracker["path_seconds"], dtype=float)
            tracker_runtime[method] += float(tracker["runtime_sec"])
            valid = bool(
                path.ndim == 2 and path.shape[0] == 2 and path.shape[1] >= 20
                and np.isfinite(path).all() and np.all(np.diff(path[0]) >= -1e-12)
                and np.all(path[1] >= -1e-7)
                and np.all(path[1] <= float(case["score_duration"]) + 1e-7)
                and tracker["path_valid"]
            )
            causal = tracker["prefix_causality"]
            causal_ok = bool(causal["pass"])
            if case_id.endswith("p01"):
                causal_ok = causal_ok and not causal.get("not_run", False)
                causal_ok = causal_ok and causal["same_shape"] and causal["max_abs_residual"] <= 1e-9
            else:
                causal_ok = causal_ok and causal.get("not_run", False)
            implementation_checks.append(valid and causal_ok)

            other = METHODS[1] if method == METHODS[0] else METHODS[0]
            other_path = np.asarray(case["trackers"][other]["path_seconds"], dtype=float)
            for index in burst_events(path):
                local_funnel["raw_bursts"] += 1
                funnel_by_method[method]["raw_bursts"] += 1
                t0 = float(path[0, index])
                t1 = t0 + DELAY
                if t1 > query[-1] or t1 > path[0, -1] or t1 > other_path[0, -1] or t1 > duration:
                    local_funnel["future_boundary_excluded"] += 1
                    funnel_by_method[method]["future_boundary_excluded"] += 1
                    continue
                relocated = latest(path, t1)
                other_t0 = latest(other_path, t0)
                other_t1 = latest(other_path, t1)
                if relocated is None or other_t0 is None or other_t1 is None:
                    local_funnel["lookup_excluded"] += 1
                    funnel_by_method[method]["lookup_excluded"] += 1
                    continue
                origin = float(path[1, index - 1])
                destination = float(path[1, index])
                start = max(0, int(np.floor(t0 / HOP)))
                end = min(int(np.ceil(duration / HOP)), int(np.ceil(t1 / HOP)))
                oi, di = int(np.floor(origin / HOP)), int(np.floor(destination / HOP))
                if (
                    end - start < WINDOW - 1 or oi < 0 or di < 0
                    or oi + WINDOW > score_frames or di + WINDOW > score_frames
                ):
                    local_funnel["feature_window_excluded"] += 1
                    funnel_by_method[method]["feature_window_excluded"] += 1
                    continue
                local_funnel["eligible"] += 1
                funnel_by_method[method]["eligible"] += 1
                eligible_keys.add((method, round(t0, 10), round(origin, 10), round(destination, 10)))
        if eligible_keys != expected_keys:
            raise AssertionError(f"eligible event identity mismatch: {case_id}")
        funnel_by_case[case_id] = dict(local_funnel)

    if case_events != aggregate:
        raise AssertionError("case event concatenation differs from aggregate events")
    if len({event["event_id"] for event in aggregate}) != len(aggregate):
        raise AssertionError("duplicate event id")

    max_residual = 0.0
    recomputed = []
    for event in aggregate:
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
        }
        for key, value in values.items():
            max_residual = max(max_residual, close(value, event[key]))
        expected_action = "accept" if accept <= hold else "hold"
        if event["oracle_action"] != expected_action:
            raise AssertionError("oracle action mismatch")
        recomputed.append({**event, **values})

    works = sorted({str(event["work"]) for event in recomputed})
    per_work = []
    for work in works:
        subset = [event for event in recomputed if event["work"] == work]
        per_work.append({
            "work": work,
            "events": len(subset),
            "performances": len({event["case"] for event in subset}),
            **{
                key: float(np.mean([event[key] for event in subset]))
                for key in ("global_regret", "immediate_regret", "delayed_regret")
            },
        })
    if per_work != result["per_work"]:
        raise AssertionError("per-work summary mismatch")

    differences = np.asarray(
        [row["global_regret"] - row["delayed_regret"] for row in per_work], dtype=float
    )
    ci = bootstrap_ci(differences) if len(differences) else [None, None]
    holds = [event for event in recomputed if event["delayed_prediction"] < 0]
    precision = (
        float(np.mean([event["oracle_action"] == "hold" for event in holds])) if holds else None
    )
    methods = sorted({event["method"] for event in recomputed})
    performances = sorted({event["case"] for event in recomputed})
    gates = {
        "implementation": all(implementation_checks),
        "support": len(recomputed) >= 20 and len(performances) >= 6 and len(works) >= 3
        and methods == list(METHODS),
        "work_equal_improvement": bool(len(differences) and differences.mean() > 0),
        "bootstrap_lower_positive": bool(len(differences) and ci[0] > 0),
        "no_material_work_worsening": bool(
            len(differences) and np.all(-differences <= 0.05 + 1e-12)
        ),
        "hold_precision": bool(len(holds) >= 5 and precision is not None and precision >= 0.8),
    }
    gates["pass"] = all(gates.values())
    if gates != result["gates"]:
        raise AssertionError(f"gate mismatch: {gates} != {result['gates']}")
    max_residual = max(max_residual, close(float(differences.mean()), result["work_equal_improvement"]))
    for left, right in zip(ci, result["bootstrap95"]):
        max_residual = max(max_residual, close(left, right))

    nonzero = differences[np.abs(differences) > 1e-12]
    exact = None
    if len(nonzero):
        signed = [
            np.mean(nonzero * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
        ]
        exact = float(np.mean(np.asarray(signed) >= nonzero.mean() - 1e-15))
    if exact != result["exact_sign_flip_p_one_sided"]:
        raise AssertionError("exact randomization mismatch")

    raw_total = sum(values["raw_bursts"] for values in funnel_by_method.values())
    eligible_total = sum(values["eligible"] for values in funnel_by_method.values())
    event_diagnostics = []
    delayed_mean = np.asarray(model["delayed"]["mean"], dtype=float)
    delayed_scale = np.asarray(model["delayed"]["scale"], dtype=float)
    for event in recomputed:
        standardized = (np.asarray(event["delayed"], dtype=float) - delayed_mean) / delayed_scale
        event_diagnostics.append({
            "event_id": event["event_id"],
            "oracle_action": event["oracle_action"],
            "delayed_prediction": event["delayed_prediction"],
            "prediction_margin_from_threshold": event["delayed_prediction"],
            "max_abs_training_standardized_feature": float(np.max(np.abs(standardized))),
            "features_beyond_3_training_scales": int(np.sum(np.abs(standardized) > 3.0)),
        })
    audit = {
        "status": "PASS",
        "frozen_transfer_status_reproduced": "PASS" if gates["pass"] else "FAIL",
        "failure_type": "insufficient_support_and_no_triggered_model_actions",
        "events": len(recomputed),
        "works_with_events": len(works),
        "performances_with_events": len(performances),
        "case_event_counts": case_counts,
        "raw_event_funnel": {
            "by_method": {method: dict(funnel_by_method[method]) for method in METHODS},
            "by_case": funnel_by_case,
            "raw_bursts": raw_total,
            "eligible": eligible_total,
            "eligibility_fraction": eligible_total / raw_total if raw_total else None,
        },
        "exposure": {
            "audio_seconds": total_audio_seconds,
            "audio_hours": total_audio_seconds / 3600.0,
            "eligible_events_per_audio_hour": len(recomputed) / (total_audio_seconds / 3600.0),
            "tracker_runtime_seconds": dict(tracker_runtime),
            "tracker_runtime_over_audio_duration": {
                method: tracker_runtime[method] / total_audio_seconds for method in METHODS
            },
        },
        "actions": {
            "delayed_hold_actions": len(holds),
            "delayed_accept_actions": len(recomputed) - len(holds),
            "oracle_hold_events": sum(event["oracle_action"] == "hold" for event in recomputed),
            "hold_precision": precision,
            "prediction_margin_range": [
                float(min(event["delayed_prediction"] for event in recomputed)),
                float(max(event["delayed_prediction"] for event in recomputed)),
            ] if recomputed else None,
            "event_diagnostics": event_diagnostics,
        },
        "max_numeric_residual": max_residual,
        "freeze_hash_drift": freeze_drift,
        "development_source_hash_drift": source_drift,
        "gates": gates,
    }
    (OUT / "independent_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
