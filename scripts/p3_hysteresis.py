#!/usr/bin/env python3
"""Frozen P3-H ambiguity-hysteresis development evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_lagged_accumulation as common

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/p2_asap_maxsim_20260908_135300/frame_results.csv"
OUT = ROOT / "results/p3_hysteresis"
ALPHA = 1.0
BOOTSTRAPS = 2000
PERMUTATIONS = 200
SEED = 20260920


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landmarks(performances):
    output = []
    for performance in performances:
        values = np.asarray(performance["rows"], dtype=float)
        for i in range(20, len(values), 20):
            required = values[i - 20:i + 1, 1:]
            if not np.all(np.isfinite(required)) or np.max(values[i - 20:i - 14, 1]) > 1.0:
                continue
            initial_times = values[i - 20:i - 14, 0] - values[i, 0]
            initial_gap = values[i - 20:i - 14, 3]
            output.append({
                "composition": performance["composition"],
                "performance_id": performance["performance_id"],
                "time_sec": float(values[i, 0]),
                "initial_gap": float(np.mean(values[i - 20:i - 15, 3])),
                "initial_gap_slope": float(np.polyfit(initial_times, initial_gap, 1)[0]),
                "current_ambiguity": float(np.mean(values[i - 4:i + 1, 4])),
                "recent_exposure": float(np.mean(values[i - 15:i - 4, 4])),
                "current_gap": float(np.mean(values[i - 4:i + 1, 3])),
            })
    return output


def arrays(rows, exposure_override=None):
    baseline = np.asarray([[r["initial_gap"], r["initial_gap_slope"], r["current_ambiguity"]]
                           for r in rows], dtype=float)
    exposure = (np.asarray([r["recent_exposure"] for r in rows], dtype=float)
                if exposure_override is None else exposure_override)
    full = np.column_stack([baseline, exposure])
    target = np.asarray([r["current_gap"] for r in rows], dtype=float)
    groups = np.asarray([r["composition"] for r in rows], dtype=object)
    return baseline, full, target, groups


def evaluate(rows, exposure_override=None, details=False):
    baseline, full, target, groups = arrays(rows, exposure_override)
    fold_rows = []
    base_prediction = np.empty(len(rows))
    full_prediction = np.empty(len(rows))
    coefficients = []
    for composition in sorted(set(groups)):
        test = groups == composition
        train = ~test
        base_model = common.fit_ridge(baseline[train], target[train], groups[train])
        full_model = common.fit_ridge(full[train], target[train], groups[train])
        base_prediction[test] = common.predict(base_model, baseline[test])
        full_prediction[test] = common.predict(full_model, full[test])
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
            "exposure_coefficient": coefficients[-1],
        })
    base_mae = float(np.mean([r["baseline_mae"] for r in fold_rows]))
    full_mae = float(np.mean([r["full_mae"] for r in fold_rows]))
    base_rmse = float(np.mean([r["baseline_rmse"] for r in fold_rows]))
    full_rmse = float(np.mean([r["full_rmse"] for r in fold_rows]))
    result = {
        "baseline_mae": base_mae,
        "full_mae": full_mae,
        "mae_relative_improvement": (base_mae - full_mae) / base_mae,
        "baseline_rmse": base_rmse,
        "full_rmse": full_rmse,
        "rmse_relative_improvement": (base_rmse - full_rmse) / base_rmse,
        "composition_mae_improved_share": float(np.mean([r["full_mae"] < r["baseline_mae"] for r in fold_rows])),
        "positive_coefficient_share": float(np.mean(np.asarray(coefficients) > 0)),
    }
    if details:
        result["folds"] = fold_rows
        result["predictions"] = (base_prediction, full_prediction)
    return result


def shifted_exposure(rows, rng):
    values = np.asarray([r["recent_exposure"] for r in rows], dtype=float)
    shifted = values.copy()
    by_performance = {}
    for index, row in enumerate(rows):
        by_performance.setdefault(row["performance_id"], []).append(index)
    for indices in by_performance.values():
        n = len(indices)
        if n < 2:
            continue
        minimum = min(10, max(1, n // 2))
        allowed = [s for s in range(1, n) if min(s, n - s) >= minimum] or list(range(1, n))
        shifted[indices] = np.roll(values[indices], int(rng.choice(allowed)))
    return shifted


def bootstrap_interval(folds, rng):
    differences = np.asarray([r["baseline_mae"] - r["full_mae"] for r in folds])
    baselines = np.asarray([r["baseline_mae"] for r in folds])
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
    performances = common.load_performances(SOURCE)
    rows = landmarks(performances)
    observed = evaluate(rows, details=True)
    rng = np.random.default_rng(SEED)
    interval = bootstrap_interval(observed["folds"], rng)
    null = [evaluate(rows, shifted_exposure(rows, rng))["mae_relative_improvement"]
            for _ in range(PERMUTATIONS)]
    permutation_p = (1 + sum(v >= observed["mae_relative_improvement"] for v in null)) / (PERMUTATIONS + 1)
    compositions = len({r["composition"] for r in rows})
    represented_performances = len({r["performance_id"] for r in rows})
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
        "hypothesis": "P3-H ambiguity hysteresis",
        "scope": "Already-open ASAP P2 development; not confirmation",
        "parameters": {"landmark_step_sec": 2.0, "pre_exposure_health_window_sec": [-2.0, -1.5],
                       "health_threshold_sec": 1.0, "recent_exposure_window_sec": [-1.5, -0.5],
                       "current_window_sec": [-0.4, 0.0], "ridge_alpha": ALPHA,
                       "bootstraps": BOOTSTRAPS, "permutations": PERMUTATIONS, "seed": SEED},
        "counts": {"source_performances": len(performances), "represented_performances": represented_performances,
                   "compositions": compositions, "landmarks": len(rows)},
        "metrics": {**observed, "mae_improvement_bootstrap_95": interval, "permutation_p": permutation_p,
                    "null_improvement_p95": float(np.percentile(null, 95))},
        "gates": gates,
        "development_pass": all(gates.values()),
        "external_validation_authorized": all(gates.values()),
        "p3_confirmed": False,
        "provenance": {"source_sha256": digest(SOURCE), "protocol_sha256": digest(ROOT / "P3_HYSTERESIS_PROTOCOL.md"),
                       "script_sha256": digest(Path(__file__))},
    }
    write_csv(OUT / "landmarks.csv", list(rows[0]), rows)
    write_csv(OUT / "folds.csv", list(folds[0]), folds)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
