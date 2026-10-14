"""One-shot frozen relocation-validator evaluation on the preserved lockbox."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from matchmaker_acquisition_smoke import run_tracker, score_clock  # noqa: E402
from matchmaker_ambiguity_rescue import predict_ridge  # noqa: E402
from online_relocation_validation import make_events  # noqa: E402


MANIFEST = ROOT/"output"/"ambiguity_map_fresh_cohort"/"manifest.csv"
ANNOTATIONS = ROOT/"asap_dataset"/"asap_annotations.json"
PROTOCOL = ROOT/"ONLINE_RELOCATION_LOCKBOX_PROTOCOL.md"
FROZEN = ROOT/"output"/"online_relocation_validator_frozen"/"model.json"
OUT = ROOT/"results"/"online_relocation_validation_lockbox"
METHODS = ("arzt", "outerhmm")
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == "lockbox"]
    if len(rows) != 8 or [row["case_id"] for row in rows] != [f"fl{i:03d}" for i in range(8)]:
        raise ValueError("Expected frozen fl000-fl007 lockbox")
    return rows


def model_arrays(value: dict[str, object]) -> dict[str, np.ndarray]:
    return {key: np.asarray(array, float) for key, array in value.items()}


def ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(samples.mean(axis=1), [.025, .975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    rows = read_rows()
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    model = json.loads(FROZEN.read_text(encoding="utf-8"))
    for relative, expected in model["source_hashes"].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f"Frozen development source drift: {relative}")
    input_files = [Path(__file__), PROTOCOL, FROZEN, MANIFEST, ANNOTATIONS, ROOT/"scripts"/"online_relocation_validation.py"]
    input_files += [ROOT/"asap_dataset"/row[key] for row in rows for key in ("score", "performance")]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in input_files}
    OUT.mkdir(parents=True)
    write_json(OUT/"freeze.json", freeze)
    write_json(OUT/"state.json", {"status": "RUNNING"})
    all_events = []
    with threadpool_limits(limits=2):
        for number, row in enumerate(rows, 1):
            print(f"[{number}/8] {row['case_id']} {row['work_family']}", flush=True)
            score = ROOT/"asap_dataset"/row["score"]
            performance = ROOT/"asap_dataset"/row["performance"]
            score_beats, score_seconds, clock_audit = score_clock(score)
            trackers = {}
            for method in METHODS:
                path = run_tracker(score, performance, method)
                path[1] = np.interp(path[1], score_beats, score_seconds)
                trackers[method] = {"path": path.tolist()}
                print(f"  {method}: path={path.shape[1]}", flush=True)
            annotation = annotations[row["performance"]]
            query = np.asarray(annotation["performance_beats"], float)
            truth = np.asarray(annotation["midi_score_beats"], float)
            order = np.argsort(query)
            case = {
                "case": row["case_id"],
                "work": row["work_family"],
                "score": row["score"],
                "performance": row["performance"],
                "query_time": query[order].tolist(),
                "truth": truth[order].tolist(),
                "trackers": trackers,
                "score_clock_audit": clock_audit,
            }
            events = make_events(case)
            all_events.extend(events)
            write_json(OUT/f"{row['case_id']}_events.json", events)
            print(f"  eligible_events={len(events)}", flush=True)
    immediate_model = model_arrays(model["immediate"])
    delayed_model = model_arrays(model["delayed"])
    for event in all_events:
        pi = float(predict_ridge(np.asarray([event["immediate"]], float), immediate_model)[0])
        pd = float(predict_ridge(np.asarray([event["delayed"]], float), delayed_model)[0])
        hold, accept = event["hold_loss"], event["accept_loss"]
        oracle = min(hold, accept)
        event["immediate_prediction"] = pi
        event["delayed_prediction"] = pd
        event["global_regret"] = float(accept-oracle)
        event["immediate_regret"] = float((accept if pi >= 0 else hold)-oracle)
        event["delayed_regret"] = float((accept if pd >= 0 else hold)-oracle)
        event["global_correct"] = bool(accept <= hold)
        event["immediate_correct"] = bool((pi >= 0) == (accept <= hold))
        event["delayed_correct"] = bool((pd >= 0) == (accept <= hold))
    write_json(OUT/"events.json", all_events)
    works = sorted(set(event["work"] for event in all_events))
    per_work = []
    for work in works:
        events = [event for event in all_events if event["work"] == work]
        per_work.append(
            {
                "work": work,
                "events": len(events),
                "global_regret": float(np.mean([event["global_regret"] for event in events])),
                "immediate_regret": float(np.mean([event["immediate_regret"] for event in events])),
                "delayed_regret": float(np.mean([event["delayed_regret"] for event in events])),
                "global_accuracy": float(np.mean([event["global_correct"] for event in events])),
                "immediate_accuracy": float(np.mean([event["immediate_correct"] for event in events])),
                "delayed_accuracy": float(np.mean([event["delayed_correct"] for event in events])),
            }
        )
    global_regret = np.array([row["global_regret"] for row in per_work])
    immediate_regret = np.array([row["immediate_regret"] for row in per_work])
    delayed_regret = np.array([row["delayed_regret"] for row in per_work])
    dg = global_regret-delayed_regret
    di = immediate_regret-delayed_regret
    actionable = global_regret > 1e-12
    floor = ~actionable
    actionable_count = int(actionable.sum())
    improved_actionable = int(np.sum(dg[actionable] > 0))
    required_actionable = int(np.ceil(.6*actionable_count)) if actionable_count else 0
    worsened_margin = int(np.sum(delayed_regret-global_regret > .05))
    gates = {
        "support": bool(len(all_events) >= 40 and len(works) >= 6 and sorted(set(event["method"] for event in all_events)) == list(METHODS)),
        "global_improvement_ci": bool(dg.mean() > 0 and ci(dg)[0] > 0),
        "immediate_improvement_ci": bool(di.mean() > 0 and ci(di)[0] > 0),
        "actionable_works": bool(actionable_count >= 3 and improved_actionable >= required_actionable),
        "floor_noninferiority": bool(np.all(delayed_regret[floor] <= .05+1e-12)),
        "no_materially_worsened_work": bool(worsened_margin == 0),
    }
    gates["pass"] = bool(all(gates.values()))
    result = {
        "status": "PASS" if gates["pass"] else "FAIL",
        "eligible_events": len(all_events),
        "works": len(works),
        "methods": sorted(set(event["method"] for event in all_events)),
        "model_sha256": digest(FROZEN),
        "work_equal": {
            "global_regret": float(global_regret.mean()),
            "immediate_regret": float(immediate_regret.mean()),
            "delayed_regret": float(delayed_regret.mean()),
            "global_minus_delayed": float(dg.mean()),
            "global_minus_delayed_ci": ci(dg),
            "immediate_minus_delayed": float(di.mean()),
            "immediate_minus_delayed_ci": ci(di),
            "global_accuracy": float(np.mean([row["global_accuracy"] for row in per_work])),
            "immediate_accuracy": float(np.mean([row["immediate_accuracy"] for row in per_work])),
            "delayed_accuracy": float(np.mean([row["delayed_accuracy"] for row in per_work])),
        },
        "floor_aware": {
            "actionable_works": actionable_count,
            "improved_actionable_works": improved_actionable,
            "required_improved_actionable_works": required_actionable,
            "floor_works": int(floor.sum()),
            "floor_noninferior_works": int(np.sum(delayed_regret[floor] <= .05+1e-12)),
            "materially_worsened_works": worsened_margin,
        },
        "per_work": per_work,
        "gates": gates,
    }
    if any(digest(ROOT/relative) != expected for relative, expected in freeze.items()):
        raise RuntimeError("Lockbox input drift")
    write_json(OUT/"result.json", result)
    write_json(OUT/"state.json", {"status": "COMPLETE", "pass": bool(gates["pass"])})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
