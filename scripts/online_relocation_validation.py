"""Development-only accept-vs-hold validation for large tracker relocations."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))

from ambiguity_induced_relocation import burst_events, frame_index  # noqa: E402
from matchmaker_ambiguity_rescue import fit_ridge, predict_ridge  # noqa: E402
from score_following.features import load_notes, notes_to_chroma_stream  # noqa: E402


SOURCE = ROOT/"results"/"matchmaker_ambiguity_rescue_development"
PROTOCOL = ROOT/"ONLINE_RELOCATION_VALIDATION_PROTOCOL.md"
OUT = ROOT/"results"/"online_relocation_validation"
METHODS = ("arzt", "outerhmm")
DELAY = 2.0
CAP = 5.0
HOP = .1
WINDOW = 20
SEED = 20260924


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def latest(path: np.ndarray, time_point: float) -> float | None:
    index = int(np.searchsorted(path[0], time_point, side="right")-1)
    return float(path[1, index]) if index >= 0 else None


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first)*np.linalg.norm(second)
    return float(np.dot(first, second)/denominator) if denominator > 0 else 0.0


def observation_support(
    performance_chroma: np.ndarray,
    score_chroma: np.ndarray,
    t0: float,
    origin: float,
    destination: float,
) -> tuple[float, float] | None:
    performance_start = max(0, int(np.floor(t0/HOP)))
    performance_end = min(len(performance_chroma), int(np.ceil((t0+DELAY)/HOP)))
    origin_index = frame_index(origin, len(score_chroma)-WINDOW+1)
    destination_index = frame_index(destination, len(score_chroma)-WINDOW+1)
    if performance_end-performance_start < WINDOW-1 or origin_index is None or destination_index is None:
        return None
    observed = performance_chroma[performance_start:performance_end].mean(axis=0)
    origin_score = score_chroma[origin_index:origin_index+WINDOW].mean(axis=0)
    destination_score = score_chroma[destination_index:destination_index+WINDOW].mean(axis=0)
    return cosine(observed, origin_score), cosine(observed, destination_score)


def make_events(case: dict[str, object]) -> list[dict[str, object]]:
    score_path = ROOT/"asap_dataset"/case["score"]
    performance_path = ROOT/"asap_dataset"/case["performance"]
    score_notes = load_notes(score_path)
    performance_notes = load_notes(performance_path)
    score_duration = max(note[1] for note in score_notes)
    performance_duration = max(note[1] for note in performance_notes)
    score_chroma = notes_to_chroma_stream(score_notes, score_duration, HOP)
    performance_chroma = notes_to_chroma_stream(performance_notes, performance_duration, HOP)
    truth_time = np.asarray(case["query_time"], float)
    truth_position = np.asarray(case["truth"], float)
    paths = {method: np.asarray(case["trackers"][method]["path"], float) for method in METHODS}
    events = []
    for method in METHODS:
        other = METHODS[1] if method == METHODS[0] else METHODS[0]
        path, other_path = paths[method], paths[other]
        for event_index in burst_events(path):
            t0 = float(path[0, event_index])
            t1 = t0+DELAY
            if t1 > truth_time[-1] or t1 > path[0, -1] or t1 > other_path[0, -1] or t1 > performance_duration:
                continue
            origin, destination = float(path[1, event_index-1]), float(path[1, event_index])
            relocated_t1 = latest(path, t1)
            other_t0, other_t1 = latest(other_path, t0), latest(other_path, t1)
            if relocated_t1 is None or other_t0 is None or other_t1 is None:
                continue
            support = observation_support(performance_chroma, score_chroma, t0, origin, destination)
            if support is None:
                continue
            origin_support, destination_support = support
            truth_t1 = float(np.interp(t1, truth_time, truth_position))
            hold_loss = min(abs(origin-truth_t1), CAP)
            accept_loss = min(abs(relocated_t1-truth_t1), CAP)
            immediate = np.array(
                [
                    np.log1p(abs(destination-origin)),
                    np.sign(destination-origin),
                    origin/max(score_duration, 1e-12),
                    destination/max(score_duration, 1e-12),
                    min(abs(other_t0-origin), 10.0),
                    min(abs(other_t0-destination), 10.0),
                    np.clip(abs(other_t0-origin)-abs(other_t0-destination), -10, 10),
                ],
                float,
            )
            delayed = np.concatenate(
                [
                    immediate,
                    np.array(
                        [
                            min(abs(relocated_t1-origin), 10.0),
                            min(abs(relocated_t1-destination), 10.0),
                            min(abs(other_t1-origin), 10.0),
                            min(abs(other_t1-destination), 10.0),
                            min(abs(relocated_t1-other_t1), 10.0),
                            np.clip(abs(other_t1-origin)-abs(other_t1-destination), -10, 10),
                            min(abs(relocated_t1-destination), 10.0),
                            origin_support,
                            destination_support,
                            destination_support-origin_support,
                        ],
                        float,
                    ),
                ]
            )
            events.append(
                {
                    "event_id": f"{case['case']}_{method}_v{len(events)+1:03d}",
                    "case": case["case"],
                    "work": case["work"],
                    "method": method,
                    "t0": t0,
                    "t1": t1,
                    "origin": origin,
                    "destination": destination,
                    "relocated_t1": relocated_t1,
                    "other_t0": other_t0,
                    "other_t1": other_t1,
                    "truth_t1": truth_t1,
                    "hold_loss": float(hold_loss),
                    "accept_loss": float(accept_loss),
                    "target": float(hold_loss-accept_loss),
                    "oracle_action": "accept" if accept_loss <= hold_loss else "hold",
                    "immediate": immediate.tolist(),
                    "delayed": delayed.tolist(),
                }
            )
    return events


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(samples.mean(axis=1), [.025, .975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    case_files = [SOURCE/f"fd{i:03d}.json" for i in range(8)]
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_files]
    score_and_performance = [ROOT/"asap_dataset"/case[key] for case in cases for key in ("score", "performance")]
    frozen = [Path(__file__), PROTOCOL, SOURCE/"result.json", *case_files, *score_and_performance]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in frozen}
    OUT.mkdir(parents=True)
    write_json(OUT/"freeze.json", freeze)
    write_json(OUT/"state.json", {"status": "RUNNING", "lockbox_authorized": False})
    events = [event for case in cases for event in make_events(case)]
    write_json(OUT/"events.json", events)
    works = sorted(set(event["work"] for event in events))
    fold_rows = []
    for held_work in works:
        train = [event for event in events if event["work"] != held_work]
        test = [event for event in events if event["work"] == held_work]
        groups = np.array([event["work"] for event in train])
        target = np.array([event["target"] for event in train])
        immediate_train = np.array([event["immediate"] for event in train])
        delayed_train = np.array([event["delayed"] for event in train])
        immediate_model = fit_ridge(immediate_train, target, groups)
        delayed_model = fit_ridge(delayed_train, target, groups)
        immediate_prediction = predict_ridge(np.array([event["immediate"] for event in test]), immediate_model)
        delayed_prediction = predict_ridge(np.array([event["delayed"] for event in test]), delayed_model)
        train_work_targets = [np.mean([event["target"] for event in train if event["work"] == work]) for work in sorted(set(groups))]
        global_accept = np.mean(train_work_targets) >= 0
        hold = np.array([event["hold_loss"] for event in test])
        accept = np.array([event["accept_loss"] for event in test])
        oracle = np.minimum(hold, accept)
        global_loss = accept if global_accept else hold
        immediate_loss = np.where(immediate_prediction >= 0, accept, hold)
        delayed_loss = np.where(delayed_prediction >= 0, accept, hold)
        fold_rows.append(
            {
                "work": held_work,
                "events": len(test),
                "global_action": "accept" if global_accept else "hold",
                "global_regret": float(np.mean(global_loss-oracle)),
                "immediate_regret": float(np.mean(immediate_loss-oracle)),
                "delayed_regret": float(np.mean(delayed_loss-oracle)),
                "global_accuracy": float(np.mean((global_accept & (accept <= hold)) | ((not global_accept) & (hold < accept)))),
                "immediate_accuracy": float(np.mean((immediate_prediction >= 0) == (accept <= hold))),
                "delayed_accuracy": float(np.mean((delayed_prediction >= 0) == (accept <= hold))),
            }
        )
    global_regret = np.array([row["global_regret"] for row in fold_rows])
    immediate_regret = np.array([row["immediate_regret"] for row in fold_rows])
    delayed_regret = np.array([row["delayed_regret"] for row in fold_rows])
    improvement_immediate = immediate_regret-delayed_regret
    improvement_global = global_regret-delayed_regret
    required = int(np.ceil(.6*len(works)))
    methods = sorted(set(event["method"] for event in events))
    gates = {
        "support": bool(len(events) >= 40 and len(works) >= 6 and methods == list(METHODS)),
        "beats_immediate_ci": bool(improvement_immediate.mean() > 0 and bootstrap_ci(improvement_immediate)[0] > 0),
        "beats_global_ci": bool(improvement_global.mean() > 0 and bootstrap_ci(improvement_global)[0] > 0),
        "positive_works": bool(np.sum(improvement_immediate > 0) >= required and np.sum(improvement_global > 0) >= required),
    }
    gates["pass"] = bool(all(gates.values()))
    result = {
        "eligible_events": len(events),
        "works": len(works),
        "methods": methods,
        "oracle_actions": dict(sorted(Counter(event["oracle_action"] for event in events).items())),
        "work_equal": {
            "global_regret": float(global_regret.mean()),
            "immediate_regret": float(immediate_regret.mean()),
            "delayed_regret": float(delayed_regret.mean()),
            "immediate_minus_delayed": float(improvement_immediate.mean()),
            "immediate_minus_delayed_ci": bootstrap_ci(improvement_immediate),
            "global_minus_delayed": float(improvement_global.mean()),
            "global_minus_delayed_ci": bootstrap_ci(improvement_global),
            "delayed_better_than_immediate_works": int(np.sum(improvement_immediate > 0)),
            "delayed_better_than_global_works": int(np.sum(improvement_global > 0)),
            "required_positive_works": required,
            "global_accuracy": float(np.mean([row["global_accuracy"] for row in fold_rows])),
            "immediate_accuracy": float(np.mean([row["immediate_accuracy"] for row in fold_rows])),
            "delayed_accuracy": float(np.mean([row["delayed_accuracy"] for row in fold_rows])),
        },
        "folds": fold_rows,
        "gates": gates,
        "lockbox_authorized": bool(gates["pass"]),
    }
    if any(sha256(ROOT/relative) != expected for relative, expected in freeze.items()):
        raise RuntimeError("Frozen input changed")
    write_json(OUT/"result.json", result)
    write_json(OUT/"state.json", {"status": "COMPLETE", "lockbox_authorized": bool(gates["pass"])})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
