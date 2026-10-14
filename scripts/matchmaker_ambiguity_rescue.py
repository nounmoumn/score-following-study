"""Development-only Matchmaker pair screen and deployable ambiguity-map test."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from matchmaker_acquisition_smoke import run_tracker, score_clock  # noqa: E402
from score_following.confidence_audit import blocked_maxsim  # noqa: E402
from score_following.features import load_notes, notes_to_chroma_stream  # noqa: E402


MANIFEST = ROOT / "output" / "ambiguity_map_fresh_cohort" / "manifest.csv"
ANNOTATIONS = ROOT / "asap_dataset" / "asap_annotations.json"
PROTOCOL = ROOT / "MATCHMAKER_AMBIGUITY_RESCUE_PROTOCOL.md"
SMOKE = ROOT / "results" / "matchmaker_acquisition_smoke" / "result.json"
OUT = ROOT / "results" / "matchmaker_ambiguity_rescue_development"
CAP = 5.0
ACCURACY_TOLERANCE = 0.5
RIDGE_ALPHA = 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["role"] == "development"]
    if len(rows) != 8 or [row["case_id"] for row in rows] != [
        f"fd{i:03d}" for i in range(8)
    ]:
        raise ValueError("Expected exactly fd000-fd007 development rows")
    return rows


def zoh(path: np.ndarray, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(path[0], query_times, side="right") - 1
    present = indices >= 0
    clipped = np.maximum(indices, 0)
    values = np.full(len(query_times), np.nan, dtype=float)
    update_times = np.full(len(query_times), np.nan, dtype=float)
    values[present] = path[1, clipped[present]]
    update_times[present] = path[0, clipped[present]]
    return values, update_times


def ambiguity_curve(score_midi: Path) -> tuple[np.ndarray, np.ndarray]:
    notes = load_notes(score_midi)
    duration = max(note[1] for note in notes)
    chroma = notes_to_chroma_stream(notes, duration, 0.1)
    ambiguity, _ = blocked_maxsim(chroma, window=20, exclude=20, block=256)
    clock = (np.arange(len(ambiguity), dtype=float) + 1.0) * 0.1
    finite = np.isfinite(ambiguity)
    if finite.sum() < 2:
        raise ValueError("Ambiguity curve has fewer than two finite points")
    # Boundary NaNs are unavailable score locations, not evidence of low ambiguity.
    return clock[finite], np.asarray(ambiguity, float)[finite]


def evaluate_case(
    row: dict[str, str], methods: list[str], annotations: dict[str, object]
) -> dict[str, object]:
    score = ROOT / "asap_dataset" / row["score"]
    performance = ROOT / "asap_dataset" / row["performance"]
    score_beats, score_seconds, clock_audit = score_clock(score)
    annotation = annotations[row["performance"]]
    if not annotation.get("score_and_performance_aligned"):
        raise ValueError(f"Unaligned ASAP case {row['case_id']}")
    query = np.asarray(annotation["performance_beats"], float)
    truth = np.asarray(annotation["midi_score_beats"], float)
    order = np.argsort(query)
    query, truth = query[order], truth[order]
    keep = np.isfinite(query) & np.isfinite(truth) & (query >= 0)
    query, truth = query[keep], truth[keep]
    if len(query) < 20:
        raise ValueError(f"Too few GT beats for {row['case_id']}")

    trackers: dict[str, object] = {}
    for method in methods:
        started = time.perf_counter()
        path_beats = run_tracker(score, performance, method)
        path_seconds = path_beats.copy()
        path_seconds[1] = np.interp(path_beats[1], score_beats, score_seconds)
        position, update_time = zoh(path_seconds, query)
        raw_error = np.abs(position - truth)
        raw_error[~np.isfinite(raw_error)] = np.inf
        loss = np.minimum(raw_error, CAP)
        trackers[method] = {
            "path": path_seconds.tolist(),
            "position": position.tolist(),
            "update_time": update_time.tolist(),
            "loss": loss.tolist(),
            "accuracy": float(np.mean(raw_error <= ACCURACY_TOLERANCE)),
            "capped_loss": float(np.mean(loss)),
            "first_output_sec": float(path_seconds[0, 0]),
            "path_points": int(path_seconds.shape[1]),
            "runtime_sec": float(time.perf_counter() - started),
        }
        print(
            f"  {method}: acc={trackers[method]['accuracy']:.4f} "
            f"loss={trackers[method]['capped_loss']:.4f} "
            f"path={path_seconds.shape[1]}",
            flush=True,
        )
    amb_clock, amb_value = ambiguity_curve(score)
    return {
        "case": row["case_id"],
        "work": row["work_family"],
        "performance": row["performance"],
        "score": row["score"],
        "query_time": query.tolist(),
        "truth": truth.tolist(),
        "score_clock_audit": clock_audit,
        "ambiguity_clock": amb_clock.tolist(),
        "ambiguity_value": amb_value.tolist(),
        "trackers": trackers,
    }


def screen_pairs(cases: list[dict[str, object]], methods: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {}
    per_work_loss: dict[str, np.ndarray] = {}
    for method in methods:
        losses = np.array([case["trackers"][method]["capped_loss"] for case in cases])
        accuracy = np.array([case["trackers"][method]["accuracy"] for case in cases])
        per_work_loss[method] = losses
        summary[method] = {
            "work_equal_accuracy": float(accuracy.mean()),
            "work_equal_capped_loss": float(losses.mean()),
            "per_work_accuracy": accuracy.tolist(),
            "per_work_capped_loss": losses.tolist(),
        }

    checks = []
    for first, second in combinations(methods, 2):
        first_loss, second_loss = per_work_loss[first], per_work_loss[second]
        first_mean = summary[first]["work_equal_capped_loss"]
        second_mean = summary[second]["work_equal_capped_loss"]
        ratios = max(first_mean, second_mean) / max(min(first_mean, second_mean), 1e-12)
        accuracy_gap = abs(
            summary[first]["work_equal_accuracy"] - summary[second]["work_equal_accuracy"]
        )
        first_wins = int(np.sum(first_loss < second_loss))
        second_wins = int(np.sum(second_loss < first_loss))
        disagreement_by_work = []
        for case in cases:
            left = np.asarray(case["trackers"][first]["position"], float)
            right = np.asarray(case["trackers"][second]["position"], float)
            available = np.isfinite(left) & np.isfinite(right)
            disagreement_by_work.append(
                float(np.mean(np.abs(left[available] - right[available]) > 0.5))
                if available.any()
                else 0.0
            )
        disagreement = float(np.mean(disagreement_by_work))
        passed = bool(
            accuracy_gap <= 0.03
            and ratios <= 1.15
            and first_wins >= 2
            and second_wins >= 2
            and disagreement >= 0.05
        )
        checks.append(
            {
                "pair": [first, second],
                "accuracy_gap": float(accuracy_gap),
                "loss_ratio": float(ratios),
                "work_wins": [first_wins, second_wins],
                "map_disagreement": disagreement,
                "per_work_disagreement": disagreement_by_work,
                "pass": passed,
            }
        )
    eligible = [check for check in checks if check["pass"]]
    eligible.sort(
        key=lambda check: (
            -check["map_disagreement"],
            abs(check["loss_ratio"] - 1.0),
            tuple(check["pair"]),
        )
    )
    return {
        "systems": summary,
        "pair_checks": checks,
        "selected_pair": eligible[0]["pair"] if eligible else None,
    }


def rolling_state(position: np.ndarray, update: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    delta = np.diff(position, prepend=position[0])
    features = np.zeros((len(position), 4), dtype=float)
    for index in range(len(position)):
        recent = delta[max(0, index - 2) : index + 1]
        features[index] = [
            delta[index],
            time_axis[index] - update[index],
            float(np.mean(recent)),
            float(np.std(recent)),
        ]
    return features


def map_rows(case: dict[str, object], pair: list[str]) -> dict[str, object]:
    first, second = pair
    time_axis = np.asarray(case["query_time"], float)
    first_tracker = case["trackers"][first]
    second_tracker = case["trackers"][second]
    first_position = np.asarray(first_tracker["position"], float)
    second_position = np.asarray(second_tracker["position"], float)
    first_update = np.asarray(first_tracker["update_time"], float)
    second_update = np.asarray(second_tracker["update_time"], float)
    usable = (
        np.isfinite(first_position)
        & np.isfinite(second_position)
        & np.isfinite(first_update)
        & np.isfinite(second_update)
    )
    time_axis = time_axis[usable]
    first_position = first_position[usable]
    second_position = second_position[usable]
    first_update = first_update[usable]
    second_update = second_update[usable]
    first_loss = np.asarray(first_tracker["loss"], float)[usable]
    second_loss = np.asarray(second_tracker["loss"], float)[usable]
    state = np.column_stack(
        [
            first_position,
            second_position,
            np.abs(first_position - second_position),
            rolling_state(first_position, first_update, time_axis),
            rolling_state(second_position, second_update, time_axis),
        ]
    )
    amb_clock = np.asarray(case["ambiguity_clock"], float)
    amb_value = np.asarray(case["ambiguity_value"], float)
    first_ambiguity = np.interp(first_position, amb_clock, amb_value)
    second_ambiguity = np.interp(second_position, amb_clock, amb_value)
    ambiguity = np.column_stack(
        [
            first_ambiguity,
            second_ambiguity,
            np.minimum(first_ambiguity, second_ambiguity),
            np.maximum(first_ambiguity, second_ambiguity),
            np.abs(first_ambiguity - second_ambiguity),
        ]
    )
    return {
        "case": case["case"],
        "work": case["work"],
        "state": state,
        "state_ambiguity": np.column_stack([state, ambiguity]),
        "target": first_loss - second_loss,
        "first_loss": first_loss,
        "second_loss": second_loss,
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    unique = np.unique(groups)
    weights = np.array([1.0 / np.sum(groups == group) for group in groups], float)
    weights /= len(unique)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-12)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights * y),
    )
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def predict_ridge(x: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    return model["coefficient"][0] + (
        (x - model["mean"]) / model["scale"]
    ) @ model["coefficient"][1:]


def cross_validate(cases: list[dict[str, object]], pair: list[str]) -> dict[str, object]:
    mapped = [map_rows(case, pair) for case in cases]
    folds = []
    for held_out in mapped:
        training = [case for case in mapped if case["work"] != held_out["work"]]
        groups = np.concatenate(
            [np.repeat(case["work"], len(case["target"])) for case in training]
        )
        y_train = np.concatenate([case["target"] for case in training])
        state_train = np.row_stack([case["state"] for case in training])
        ambiguity_train = np.row_stack([case["state_ambiguity"] for case in training])
        state_model = fit_ridge(state_train, y_train, groups)
        ambiguity_model = fit_ridge(ambiguity_train, y_train, groups)
        state_prediction = predict_ridge(held_out["state"], state_model)
        ambiguity_prediction = predict_ridge(
            held_out["state_ambiguity"], ambiguity_model
        )
        target = held_out["target"]
        first_loss = held_out["first_loss"]
        second_loss = held_out["second_loss"]
        oracle = np.minimum(first_loss, second_loss)
        train_first = np.mean([np.mean(case["first_loss"]) for case in training])
        train_second = np.mean([np.mean(case["second_loss"]) for case in training])
        global_loss = first_loss if train_first <= train_second else second_loss
        state_selected = np.where(state_prediction >= 0, second_loss, first_loss)
        ambiguity_selected = np.where(
            ambiguity_prediction >= 0, second_loss, first_loss
        )
        folds.append(
            {
                "work": held_out["work"],
                "rows": int(len(target)),
                "state_mse": float(np.mean((target - state_prediction) ** 2)),
                "state_ambiguity_mse": float(
                    np.mean((target - ambiguity_prediction) ** 2)
                ),
                "global_choice": pair[0] if train_first <= train_second else pair[1],
                "global_regret": float(np.mean(global_loss - oracle)),
                "state_regret": float(np.mean(state_selected - oracle)),
                "state_ambiguity_regret": float(
                    np.mean(ambiguity_selected - oracle)
                ),
            }
        )
    state_mse = np.array([fold["state_mse"] for fold in folds])
    ambiguity_mse = np.array([fold["state_ambiguity_mse"] for fold in folds])
    global_regret = np.array([fold["global_regret"] for fold in folds])
    state_regret = np.array([fold["state_regret"] for fold in folds])
    ambiguity_regret = np.array([fold["state_ambiguity_regret"] for fold in folds])
    gates = {
        "mean_mse_improves": bool(ambiguity_mse.mean() < state_mse.mean()),
        "mse_improves_at_least_5_works": bool(np.sum(ambiguity_mse < state_mse) >= 5),
        "regret_below_state": bool(ambiguity_regret.mean() < state_regret.mean()),
        "regret_below_global": bool(ambiguity_regret.mean() < global_regret.mean()),
        "prefix_features_pass": True,
    }
    gates["pass"] = bool(all(gates.values()))
    return {
        "folds": folds,
        "work_equal": {
            "state_mse": float(state_mse.mean()),
            "state_ambiguity_mse": float(ambiguity_mse.mean()),
            "mse_improvement": float((state_mse - ambiguity_mse).mean()),
            "mse_positive_works": int(np.sum(ambiguity_mse < state_mse)),
            "global_regret": float(global_regret.mean()),
            "state_regret": float(state_regret.mean()),
            "state_ambiguity_regret": float(ambiguity_regret.mean()),
        },
        "gates": gates,
    }


def full_model(cases: list[dict[str, object]], pair: list[str]) -> dict[str, object]:
    mapped = [map_rows(case, pair) for case in cases]
    groups = np.concatenate(
        [np.repeat(case["work"], len(case["target"])) for case in mapped]
    )
    target = np.concatenate([case["target"] for case in mapped])
    state = np.row_stack([case["state"] for case in mapped])
    state_ambiguity = np.row_stack([case["state_ambiguity"] for case in mapped])

    def serialise(model: dict[str, np.ndarray]) -> dict[str, object]:
        return {key: value.tolist() for key, value in model.items()}

    return {
        "pair": pair,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "capped_loss(first)-capped_loss(second)",
        "state_feature_order": [
            "first_position",
            "second_position",
            "absolute_position_disagreement",
            "first_delta",
            "first_output_age",
            "first_recent3_delta_mean",
            "first_recent3_delta_std",
            "second_delta",
            "second_output_age",
            "second_recent3_delta_mean",
            "second_recent3_delta_std",
        ],
        "ambiguity_feature_order": [
            "first_position_maxsim",
            "second_position_maxsim",
            "min_position_maxsim",
            "max_position_maxsim",
            "absolute_maxsim_difference",
        ],
        "state": serialise(fit_ridge(state, target, groups)),
        "state_ambiguity": serialise(fit_ridge(state_ambiguity, target, groups)),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    if not smoke.get("development_authorized"):
        raise SystemExit("Smoke did not authorize development")
    methods = list(smoke["passing_methods"])
    rows = read_manifest()
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    files = [Path(__file__), PROTOCOL, MANIFEST, ANNOTATIONS, SMOKE]
    files += [ROOT / "scripts" / "matchmaker_acquisition_smoke.py"]
    files += [ROOT / "external" / "matchmaker" / "matchmaker" / "matchmaker.py"]
    files += [
        ROOT / "asap_dataset" / row[key]
        for row in rows
        for key in ("score", "performance")
    ]
    frozen = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in files
    }
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", frozen)
    write_json(
        OUT / "state.json",
        {"status": "RUNNING", "methods": methods, "lockbox_authorized": False},
    )
    cases = []
    with threadpool_limits(limits=2):
        for index, row in enumerate(rows, 1):
            print(f"[{index}/8] {row['case_id']} {row['work_family']}", flush=True)
            try:
                case = evaluate_case(row, methods, annotations)
            except Exception:
                write_json(
                    OUT / f"{row['case_id']}_error.json",
                    {"traceback": traceback.format_exc()},
                )
                raise
            cases.append(case)
            write_json(OUT / f"{row['case_id']}.json", case)
    screen = screen_pairs(cases, methods)
    result: dict[str, object] = {
        "methods": methods,
        **screen,
        "lockbox_authorized": False,
    }
    if screen["selected_pair"] is not None:
        map_test = cross_validate(cases, screen["selected_pair"])
        result["ambiguity_map"] = map_test
        result["lockbox_authorized"] = bool(map_test["gates"]["pass"])
        write_json(OUT / "model.json", full_model(cases, screen["selected_pair"]))
    if any(sha256(ROOT / relative) != digest for relative, digest in frozen.items()):
        raise RuntimeError("Frozen input changed during development run")
    write_json(OUT / "result.json", result)
    write_json(
        OUT / "state.json",
        {
            "status": "COMPLETE",
            "methods": methods,
            "selected_pair": screen["selected_pair"],
            "lockbox_authorized": result["lockbox_authorized"],
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
