"""Independent numeric and causal-feature audit for Matchmaker development run."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from matchmaker_acquisition_smoke import run_tracker, score_clock  # noqa: E402
from matchmaker_ambiguity_rescue import ambiguity_curve, rolling_state, zoh  # noqa: E402


DEV = ROOT / "results" / "matchmaker_ambiguity_rescue_development"
SMOKE = ROOT / "results" / "matchmaker_acquisition_smoke"
OUT = DEV / "independent_audit.json"
METHODS = ("arzt", "outerhmm", "hmm", "pthmm")
PAIR = ("arzt", "outerhmm")
SEED = 20260924


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()


def state_features(
    path_by_method: dict[str, np.ndarray], query: np.ndarray, score: Path
) -> tuple[np.ndarray, np.ndarray]:
    position = {}
    update = {}
    score_beats, score_seconds, _ = score_clock(score)
    for method, path_beats in path_by_method.items():
        path = path_beats.copy()
        path[1] = np.interp(path[1], score_beats, score_seconds)
        position[method], update[method] = zoh(path, query)
    valid = np.ones(len(query), dtype=bool)
    for method in PAIR:
        valid &= np.isfinite(position[method]) & np.isfinite(update[method])
    time_axis = query[valid]
    left, right = (position[method][valid] for method in PAIR)
    left_update, right_update = (update[method][valid] for method in PAIR)
    state = np.column_stack(
        [
            left,
            right,
            np.abs(left - right),
            rolling_state(left, left_update, time_axis),
            rolling_state(right, right_update, time_axis),
        ]
    )
    amb_clock, amb_value = ambiguity_curve(score)
    left_a = np.interp(left, amb_clock, amb_value)
    right_a = np.interp(right, amb_clock, amb_value)
    ambiguity = np.column_stack(
        [left_a, right_a, np.minimum(left_a, right_a), np.maximum(left_a, right_a), np.abs(left_a-right_a)]
    )
    return state, np.column_stack([state, ambiguity])


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    saved = json.loads((DEV / "result.json").read_text(encoding="utf-8"))
    cases = [json.loads((DEV / f"fd{i:03d}.json").read_text(encoding="utf-8")) for i in range(8)]

    summary = {}
    for method in METHODS:
        accuracy = np.array([case["trackers"][method]["accuracy"] for case in cases])
        loss = np.array([case["trackers"][method]["capped_loss"] for case in cases])
        summary[method] = {"accuracy": float(accuracy.mean()), "loss": float(loss.mean()), "work_loss": loss}
    pair_checks = []
    for first, second in combinations(METHODS, 2):
        first_loss, second_loss = summary[first]["work_loss"], summary[second]["work_loss"]
        disagreement = []
        for case in cases:
            left = np.asarray(case["trackers"][first]["position"], float)
            right = np.asarray(case["trackers"][second]["position"], float)
            valid = np.isfinite(left) & np.isfinite(right)
            disagreement.append(float(np.mean(np.abs(left[valid] - right[valid]) > 0.5)))
        ratio = max(summary[first]["loss"], summary[second]["loss"]) / min(summary[first]["loss"], summary[second]["loss"])
        record = {
            "pair": [first, second],
            "accuracy_gap": abs(summary[first]["accuracy"] - summary[second]["accuracy"]),
            "loss_ratio": ratio,
            "work_wins": [int(np.sum(first_loss < second_loss)), int(np.sum(second_loss < first_loss))],
            "map_disagreement": float(np.mean(disagreement)),
        }
        record["pass"] = bool(record["accuracy_gap"] <= .03 and ratio <= 1.15 and min(record["work_wins"]) >= 2 and record["map_disagreement"] >= .05)
        pair_checks.append(record)
    chosen = [record for record in pair_checks if record["pass"]]
    chosen.sort(key=lambda record: (-record["map_disagreement"], abs(record["loss_ratio"]-1), tuple(record["pair"])))

    saved_folds = saved["ambiguity_map"]["folds"]
    state_mse = np.array([fold["state_mse"] for fold in saved_folds])
    ambiguity_mse = np.array([fold["state_ambiguity_mse"] for fold in saved_folds])
    global_regret = np.array([fold["global_regret"] for fold in saved_folds])
    state_regret = np.array([fold["state_regret"] for fold in saved_folds])
    ambiguity_regret = np.array([fold["state_ambiguity_regret"] for fold in saved_folds])

    # Explicit feature-prefix check on the exact smoke prefixes and selected pair.
    score = ROOT / "asap_dataset" / cases[0]["score"]
    annotations = json.loads((ROOT / "asap_dataset" / "asap_annotations.json").read_text(encoding="utf-8"))
    query = np.asarray(annotations[cases[0]["performance"]]["performance_beats"], float)
    query = np.sort(query[(query >= 0) & (query <= 14.95)])
    long_paths = {method: run_tracker(score, SMOKE / "fd000_prefix_30s.mid", method) for method in PAIR}
    short_paths = {method: run_tracker(score, SMOKE / "fd000_prefix_15s.mid", method) for method in PAIR}
    long_state, long_ambiguity = state_features(long_paths, query, score)
    short_state, short_ambiguity = state_features(short_paths, query, score)
    state_equal = long_state.shape == short_state.shape and np.allclose(long_state, short_state, atol=1e-12, rtol=0)
    ambiguity_equal = long_ambiguity.shape == short_ambiguity.shape and np.allclose(long_ambiguity, short_ambiguity, atol=1e-12, rtol=0)

    freeze = json.loads((DEV / "freeze.json").read_text(encoding="utf-8"))
    drift = [relative for relative, digest in freeze.items() if sha256(ROOT / relative) != digest]
    saved_pair_rows = {tuple(record["pair"]): record for record in saved["pair_checks"]}
    residuals = []
    for record in pair_checks:
        prior = saved_pair_rows[tuple(record["pair"])]
        for key in ("accuracy_gap", "loss_ratio", "map_disagreement"):
            residuals.append(abs(record[key] - prior[key]))
    for method in METHODS:
        residuals += [
            abs(summary[method]["accuracy"] - saved["systems"][method]["work_equal_accuracy"]),
            abs(summary[method]["loss"] - saved["systems"][method]["work_equal_capped_loss"]),
        ]

    audit = {
        "status": "PASS" if not drift and state_equal and ambiguity_equal and max(residuals, default=0) <= 1e-12 else "FAIL",
        "cases": len(cases),
        "selected_pair_recomputed": chosen[0]["pair"] if chosen else None,
        "selected_pair_saved": saved["selected_pair"],
        "max_summary_residual": max(residuals, default=0.0),
        "prefix_feature_audit": {
            "query_rows": int(len(query)),
            "state_shape": list(long_state.shape),
            "state_equal": bool(state_equal),
            "state_ambiguity_shape": list(long_ambiguity.shape),
            "state_ambiguity_equal": bool(ambiguity_equal),
            "max_state_difference": float(np.max(np.abs(long_state-short_state), initial=0.0)) if long_state.shape == short_state.shape else None,
            "max_state_ambiguity_difference": float(np.max(np.abs(long_ambiguity-short_ambiguity), initial=0.0)) if long_ambiguity.shape == short_ambiguity.shape else None,
        },
        "ambiguity_map_work_bootstrap95": {
            "mse_improvement": ci(state_mse - ambiguity_mse),
            "state_minus_ambiguity_regret": ci(state_regret - ambiguity_regret),
            "global_minus_ambiguity_regret": ci(global_regret - ambiguity_regret),
        },
        "recomputed_work_equal": {
            "state_mse": float(state_mse.mean()),
            "state_ambiguity_mse": float(ambiguity_mse.mean()),
            "mse_improvement": float((state_mse-ambiguity_mse).mean()),
            "mse_positive_works": int(np.sum(ambiguity_mse < state_mse)),
            "global_regret": float(global_regret.mean()),
            "state_regret": float(state_regret.mean()),
            "state_ambiguity_regret": float(ambiguity_regret.mean()),
        },
        "hash_drift": drift,
        "lockbox_output_absent": not (ROOT / "results" / "matchmaker_ambiguity_rescue_lockbox").exists(),
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
