"""Independent model refit audit for online relocation validation."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"results"/"online_relocation_validation"
OUT = SOURCE/"independent_audit.json"
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    weights = np.array([1/(len(unique)*np.sum(groups == group)) for group in groups])
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-12)
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    coefficient = np.linalg.solve(design.T@(weights[:, None]*design)+penalty, design.T@(weights*y))
    return mean, scale, coefficient


def predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, scale, coefficient = model
    return coefficient[0]+((x-mean)/scale)@coefficient[1:]


def ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(samples.mean(axis=1), [.025, .975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    saved = json.loads((SOURCE/"result.json").read_text(encoding="utf-8"))
    events = json.loads((SOURCE/"events.json").read_text(encoding="utf-8"))
    freeze = json.loads((SOURCE/"freeze.json").read_text(encoding="utf-8"))
    works = sorted(set(event["work"] for event in events))
    folds = []
    for held in works:
        train = [event for event in events if event["work"] != held]
        test = [event for event in events if event["work"] == held]
        groups = np.array([event["work"] for event in train])
        target = np.array([event["target"] for event in train])
        immediate = fit(np.array([event["immediate"] for event in train]), target, groups)
        delayed = fit(np.array([event["delayed"] for event in train]), target, groups)
        pi = predict(np.array([event["immediate"] for event in test]), immediate)
        pd = predict(np.array([event["delayed"] for event in test]), delayed)
        train_work_mean = [np.mean([event["target"] for event in train if event["work"] == work]) for work in sorted(set(groups))]
        global_accept = np.mean(train_work_mean) >= 0
        hold = np.array([event["hold_loss"] for event in test])
        accept = np.array([event["accept_loss"] for event in test])
        oracle = np.minimum(hold, accept)
        folds.append(
            {
                "work": held,
                "global_regret": float(np.mean((accept if global_accept else hold)-oracle)),
                "immediate_regret": float(np.mean(np.where(pi >= 0, accept, hold)-oracle)),
                "delayed_regret": float(np.mean(np.where(pd >= 0, accept, hold)-oracle)),
            }
        )
    saved_by_work = {row["work"]: row for row in saved["folds"]}
    residuals = [abs(row[key]-saved_by_work[row["work"]][key]) for row in folds for key in ("global_regret", "immediate_regret", "delayed_regret")]
    global_regret = np.array([row["global_regret"] for row in folds])
    immediate_regret = np.array([row["immediate_regret"] for row in folds])
    delayed_regret = np.array([row["delayed_regret"] for row in folds])
    di = immediate_regret-delayed_regret
    dg = global_regret-delayed_regret
    required = int(np.ceil(.6*len(works)))
    methods = sorted(set(event["method"] for event in events))
    gates = {
        "support": bool(len(events) >= 40 and len(works) >= 6 and methods == ["arzt", "outerhmm"]),
        "beats_immediate_ci": bool(di.mean() > 0 and ci(di)[0] > 0),
        "beats_global_ci": bool(dg.mean() > 0 and ci(dg)[0] > 0),
        "positive_works": bool(np.sum(di > 0) >= required and np.sum(dg > 0) >= required),
    }
    gates["pass"] = bool(all(gates.values()))
    drift = [relative for relative, expected in freeze.items() if digest(ROOT/relative) != expected]
    audit = {
        "status": "PASS" if not drift and max(residuals, default=0) <= 1e-12 and gates == saved["gates"] else "FAIL",
        "eligible_events": len(events),
        "works": len(works),
        "methods": methods,
        "max_fold_residual": max(residuals, default=0.0),
        "immediate_minus_delayed_ci": ci(di),
        "global_minus_delayed_ci": ci(dg),
        "strictly_improved_works": int(np.sum(di > 0)),
        "tied_works": int(np.sum(np.abs(di) <= 1e-12)),
        "worsened_works": int(np.sum(di < -1e-12)),
        "gates_recomputed": gates,
        "hash_drift": drift,
        "lockbox_output_absent": not (ROOT/"results"/"online_relocation_validation_lockbox").exists(),
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
