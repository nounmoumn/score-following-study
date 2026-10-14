#!/usr/bin/env python3
"""Frozen P3-L dense lagged-accumulation development evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p2_asap_maxsim_20260908_135300/frame_results.csv"
OUT = ROOT / "results/p3_lagged_accumulation"
ALPHA = 1.0
BOOTSTRAPS = 2000
PERMUTATIONS = 200
SEED = 20260920


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_performances(path=SOURCE):
    performances = []
    current = []
    previous_label = None
    previous_time = -np.inf
    with Path(path).open(encoding="utf8", newline="") as handle:
        for row in csv.DictReader(handle):
            label, time = row["label"], float(row["perf_time_sec"])
            if current and (label != previous_label or time <= previous_time):
                performances.append({"composition": previous_label, "rows": current})
                current = []
            current.append((time, float(row["online_error_sec"]), float(row["offline_error_sec"]),
                            float(row["gap_sec"]), float(row["ambiguity"])))
            previous_label, previous_time = label, time
    if current:
        performances.append({"composition": previous_label, "rows": current})
    for index, performance in enumerate(performances):
        performance["performance_id"] = f"{performance['composition']}#{index:02d}"
    return performances


def landmarks(performances):
    output = []
    for performance in performances:
        values = np.asarray(performance["rows"], dtype=float)
        for i in range(10, len(values) - 20, 20):
            required = values[i - 10:i + 21, 1:]
            if not np.all(np.isfinite(required)) or np.max(values[i - 10:i + 1, 1]) > 1.0:
                continue
            times = values[i - 10:i + 1, 0] - values[i, 0]
            gap_past = values[i - 10:i + 1, 3]
            slope = float(np.polyfit(times, gap_past, 1)[0])
            output.append({
                "composition": performance["composition"],
                "performance_id": performance["performance_id"],
                "time_sec": float(values[i, 0]),
                "current_gap": float(np.mean(values[i - 4:i + 1, 3])),
                "past_gap_slope": slope,
                "ambiguity": float(np.mean(values[i - 4:i + 1, 4])),
                "future_gap": float(np.mean(values[i + 11:i + 21, 3])),
            })
    return output


def arrays(rows, ambiguity_override=None):
    baseline = np.asarray([[row["current_gap"], row["past_gap_slope"]] for row in rows], dtype=float)
    ambiguity = np.asarray([row["ambiguity"] for row in rows], dtype=float) if ambiguity_override is None else ambiguity_override
    full = np.column_stack([baseline, ambiguity])
    target = np.asarray([row["future_gap"] for row in rows], dtype=float)
    groups = np.asarray([row["composition"] for row in rows], dtype=object)
    return baseline, full, target, groups


def fit_ridge(train_x, train_y, train_groups):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (train_x - mean) / scale
    unique, counts = np.unique(train_groups, return_counts=True)
    count = dict(zip(unique, counts))
    weights = np.asarray([1.0 / count[group] for group in train_groups], dtype=float)
    weights *= len(unique) / weights.sum()
    design = np.column_stack([np.ones(len(x)), x])
    gram = design.T @ (weights[:, None] * design)
    penalty = np.eye(design.shape[1]) * ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, design.T @ (weights * train_y))
    return mean, scale, coefficients


def predict(model, x):
    mean, scale, coefficients = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coefficients


def evaluate(rows, ambiguity_override=None, details=False):
    baseline, full, target, groups = arrays(rows, ambiguity_override)
    compositions = sorted(set(groups))
    fold_rows = []
    base_prediction = np.empty(len(rows))
    full_prediction = np.empty(len(rows))
    coefficients = []
    for composition in compositions:
        test = groups == composition
        train = ~test
        base_model = fit_ridge(baseline[train], target[train], groups[train])
        full_model = fit_ridge(full[train], target[train], groups[train])
        base_prediction[test] = predict(base_model, baseline[test])
        full_prediction[test] = predict(full_model, full[test])
        coefficients.append(float(full_model[2][-1]))
        base_error = target[test] - base_prediction[test]
        full_error = target[test] - full_prediction[test]
        fold_rows.append({
            "composition": composition,
            "landmarks": int(np.sum(test)),
            "baseline_mae": float(np.mean(np.abs(base_error))),
            "full_mae": float(np.mean(np.abs(full_error))),
            "baseline_rmse": float(np.sqrt(np.mean(base_error ** 2))),
            "full_rmse": float(np.sqrt(np.mean(full_error ** 2))),
            "ambiguity_coefficient": coefficients[-1],
        })
    base_mae = float(np.mean([row["baseline_mae"] for row in fold_rows]))
    full_mae = float(np.mean([row["full_mae"] for row in fold_rows]))
    base_rmse = float(np.mean([row["baseline_rmse"] for row in fold_rows]))
    full_rmse = float(np.mean([row["full_rmse"] for row in fold_rows]))
    result = {
        "baseline_mae": base_mae,
        "full_mae": full_mae,
        "mae_relative_improvement": (base_mae - full_mae) / base_mae,
        "baseline_rmse": base_rmse,
        "full_rmse": full_rmse,
        "rmse_relative_improvement": (base_rmse - full_rmse) / base_rmse,
        "composition_mae_improved_share": float(np.mean([row["full_mae"] < row["baseline_mae"] for row in fold_rows])),
        "positive_coefficient_share": float(np.mean(np.asarray(coefficients) > 0)),
    }
    if details:
        result["folds"] = fold_rows
        result["predictions"] = (base_prediction, full_prediction)
    return result


def shifted_ambiguity(rows, rng):
    values = np.asarray([row["ambiguity"] for row in rows], dtype=float)
    shifted = values.copy()
    by_performance = {}
    for index, row in enumerate(rows):
        by_performance.setdefault(row["performance_id"], []).append(index)
    for indices in by_performance.values():
        n = len(indices)
        if n < 2:
            continue
        minimum = min(10, max(1, n // 2))
        allowed = [shift for shift in range(1, n) if min(shift, n - shift) >= minimum]
        if not allowed:
            allowed = list(range(1, n))
        shift = int(rng.choice(allowed))
        shifted[indices] = np.roll(values[indices], shift)
    return shifted


def bootstrap_interval(folds, rng):
    differences = np.asarray([row["baseline_mae"] - row["full_mae"] for row in folds])
    baselines = np.asarray([row["baseline_mae"] for row in folds])
    samples = []
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, len(folds), len(folds))
        samples.append(float(np.mean(differences[index]) / np.mean(baselines[index])))
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    performances = load_performances()
    rows = landmarks(performances)
    observed = evaluate(rows, details=True)
    rng = np.random.default_rng(SEED)
    interval = bootstrap_interval(observed["folds"], rng)
    null = []
    for _ in range(PERMUTATIONS):
        null.append(evaluate(rows, shifted_ambiguity(rows, rng))["mae_relative_improvement"])
    permutation_p = (1 + sum(value >= observed["mae_relative_improvement"] for value in null)) / (PERMUTATIONS + 1)
    compositions = len({row["composition"] for row in rows})
    represented_performances = len({row["performance_id"] for row in rows})
    gates = {
        "compositions_ge_20": compositions >= 20,
        "performances_ge_30": represented_performances >= 30,
        "landmarks_ge_500": len(rows) >= 500,
        "mae_improvement_ge_0_02": observed["mae_relative_improvement"] >= 0.02,
        "rmse_improvement_ge_0_02": observed["rmse_relative_improvement"] >= 0.02,
        "composition_mae_improved_share_ge_0_60": observed["composition_mae_improved_share"] >= 0.60,
        "positive_coefficient_share_ge_0_80": observed["positive_coefficient_share"] >= 0.80,
        "bootstrap_lower_gt_0": interval[0] > 0,
        "permutation_p_le_0_05": permutation_p <= 0.05,
    }
    base_prediction, full_prediction = observed.pop("predictions")
    folds = observed.pop("folds")
    for index, row in enumerate(rows):
        row["baseline_prediction"] = float(base_prediction[index])
        row["full_prediction"] = float(full_prediction[index])
    result = {
        "hypothesis": "P3-L lagged accumulation",
        "scope": "Already-open ASAP P2 development; not confirmation",
        "parameters": {"landmark_step_sec": 2.0, "health_history_sec": 1.0, "health_threshold_sec": 1.0,
                       "ambiguity_window_sec": 0.5, "target_window_sec": [1.1, 2.0], "ridge_alpha": ALPHA,
                       "bootstraps": BOOTSTRAPS, "permutations": PERMUTATIONS, "seed": SEED},
        "counts": {"source_performances": len(performances), "represented_performances": represented_performances,
                   "compositions": compositions, "landmarks": len(rows)},
        "metrics": {**observed, "mae_improvement_bootstrap_95": interval, "permutation_p": permutation_p,
                    "null_improvement_p95": float(np.percentile(null, 95))},
        "gates": gates,
        "development_pass": all(gates.values()),
        "external_validation_authorized": all(gates.values()),
        "p3_confirmed": False,
        "provenance": {"source_sha256": digest(SOURCE), "protocol_sha256": digest(ROOT / "P3_LAGGED_ACCUMULATION_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    write_csv(OUT / "landmarks.csv", list(rows[0]), rows)
    write_csv(OUT / "folds.csv", list(folds[0]), folds)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
