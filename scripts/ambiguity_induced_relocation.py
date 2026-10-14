"""Test whether static score ambiguity is a causal precursor of tracker relocation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_following.features import load_notes, notes_to_chroma_stream  # noqa: E402


SOURCE = ROOT / "results" / "matchmaker_ambiguity_rescue_development"
PROTOCOL = ROOT / "AMBIGUITY_INDUCED_RELOCATION_PROTOCOL.md"
OUT = ROOT / "results" / "ambiguity_induced_relocation"
METHODS = ("arzt", "outerhmm")
HOP = 0.1
WINDOW = 20
MIN_JUMP = 2.0
SEPARATION = 20
SEED = 20260924


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def score_windows(score_path: Path) -> tuple[np.ndarray, float]:
    notes = load_notes(score_path)
    duration = max(note[1] for note in notes)
    chroma = notes_to_chroma_stream(notes, duration, HOP)
    count = len(chroma)-WINDOW+1
    features = np.stack([chroma[index:index+WINDOW].reshape(-1) for index in range(count)])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features/np.where(norms > 0, norms, 1.0)
    return features, duration


def frame_index(second: float, count: int) -> int | None:
    index = int(round(second/HOP))-1
    return index if 0 <= index < count else None


def score_metrics(features: np.ndarray, origin: float, destination: float) -> dict[str, float] | None:
    origin_index = frame_index(origin, len(features))
    destination_index = frame_index(destination, len(features))
    if origin_index is None or destination_index is None:
        return None
    similarity = features @ features[origin_index]
    candidate = np.abs(np.arange(len(features))-origin_index) >= SEPARATION
    distant = similarity[candidate]
    if len(distant) == 0:
        return None
    destination_similarity = float(similarity[destination_index])
    return {
        "origin_maxsim": float(np.max(distant)),
        "destination_similarity": destination_similarity,
        "destination_percentile": float(np.mean(distant <= destination_similarity)),
    }


def burst_events(path: np.ndarray) -> list[int]:
    change = np.abs(np.diff(path[1]))
    candidates = [index for index in range(1, path.shape[1]) if path[0, index] >= 10 and change[index-1] >= MIN_JUMP]
    if not candidates:
        return []
    bursts = [[candidates[0]]]
    for index in candidates[1:]:
        if path[0, index]-path[0, bursts[-1][-1]] <= 2.0:
            bursts[-1].append(index)
        else:
            bursts.append([index])
    return [max(burst, key=lambda index: abs(path[1, index]-path[1, index-1])) for burst in bursts]


def controls(path: np.ndarray, event_times: np.ndarray, feature_count: int) -> list[int]:
    change = np.abs(np.diff(path[1]))
    output = []
    for index in range(1, path.shape[1]):
        if path[0, index] < 10 or change[index-1] >= .5:
            continue
        if np.min(np.abs(event_times-path[0, index]), initial=np.inf) < 10:
            continue
        if frame_index(float(path[1, index-1]), feature_count) is None:
            continue
        output.append(index)
    return output


def bootstrap_ci(work_values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = work_values[rng.integers(0, len(work_values), size=(20_000, len(work_values)))]
    return np.quantile(samples.mean(axis=1), [.025, .975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    case_files = [SOURCE/f"fd{i:03d}.json" for i in range(8)]
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_files]
    score_files = [ROOT/"asap_dataset"/case["score"] for case in cases]
    frozen = [Path(__file__), PROTOCOL, SOURCE/"result.json", *case_files, *score_files]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in frozen}
    OUT.mkdir(parents=True)
    write_json(OUT/"freeze.json", freeze)
    write_json(OUT/"state.json", {"status": "RUNNING", "lockbox_authorized": False})

    all_events = []
    raw_counts = defaultdict(int)
    for case, score_path in zip(cases, score_files):
        features, score_duration = score_windows(score_path)
        truth_time = np.asarray(case["query_time"], float)
        truth_position = np.asarray(case["truth"], float)
        for method in METHODS:
            path = np.asarray(case["trackers"][method]["path"], float)
            event_indices = burst_events(path)
            raw_counts[method] += len(event_indices)
            event_times = path[0, event_indices] if event_indices else np.empty(0)
            control_indices = controls(path, event_times, len(features))
            available = set(control_indices)
            method_events = []
            for event_index in event_indices:
                origin = float(path[1, event_index-1])
                destination = float(path[1, event_index])
                metrics = score_metrics(features, origin, destination)
                if metrics is None:
                    continue
                perf_progress = float(path[0, event_index]/max(path[0, -1], 1e-12))
                score_progress = origin/max(score_duration, 1e-12)
                ranked = sorted(
                    available,
                    key=lambda index: (
                        abs(path[0, index]/max(path[0, -1], 1e-12)-perf_progress)
                        + abs(path[1, index-1]/max(score_duration, 1e-12)-score_progress),
                        index,
                    ),
                )
                if not ranked:
                    continue
                control_index = ranked[0]
                distance = (
                    abs(path[0, control_index]/max(path[0, -1], 1e-12)-perf_progress)
                    + abs(path[1, control_index-1]/max(score_duration, 1e-12)-score_progress)
                )
                if distance > .25:
                    continue
                control_origin = float(path[1, control_index-1])
                control_metrics = score_metrics(features, control_origin, control_origin)
                if control_metrics is None:
                    continue
                available.remove(control_index)
                event_time = float(path[0, event_index])
                gt = float(np.interp(event_time, truth_time, truth_position))
                before = min(abs(origin-gt), 5.0)
                after = min(abs(destination-gt), 5.0)
                if after <= before-.5:
                    outcome = "correction"
                elif after >= before+.5:
                    outcome = "failure"
                else:
                    outcome = "neutral"
                method_events.append(
                    {
                        "event_id": f"{case['case']}_{method}_r{len(method_events)+1:03d}",
                        "case": case["case"],
                        "work": case["work"],
                        "method": method,
                        "performance_time": event_time,
                        "origin": origin,
                        "destination": destination,
                        "jump": destination-origin,
                        **metrics,
                        "control_time": float(path[0, control_index]),
                        "control_origin": control_origin,
                        "control_maxsim": control_metrics["origin_maxsim"],
                        "match_distance": float(distance),
                        "error_before": float(before),
                        "error_after": float(after),
                        "outcome": outcome,
                    }
                )
            all_events.extend(method_events)
    write_json(OUT/"events.json", all_events)

    by_work = defaultdict(list)
    for event in all_events:
        by_work[event["work"]].append(event)
    per_work = []
    for work, events in sorted(by_work.items()):
        differences = np.array([event["origin_maxsim"]-event["control_maxsim"] for event in events])
        event_high = np.array([event["origin_maxsim"] >= .8 for event in events])
        control_high = np.array([event["control_maxsim"] >= .8 for event in events])
        per_work.append(
            {
                "work": work,
                "events": len(events),
                "mean_maxsim_difference": float(differences.mean()),
                "event_high_ambiguity_rate": float(event_high.mean()),
                "control_high_ambiguity_rate": float(control_high.mean()),
            }
        )
    differences = np.array([row["mean_maxsim_difference"] for row in per_work])
    event_high = float(np.mean([row["event_high_ambiguity_rate"] for row in per_work])) if per_work else 0.0
    control_high = float(np.mean([row["control_high_ambiguity_rate"] for row in per_work])) if per_work else 0.0
    percentiles = np.array([event["destination_percentile"] for event in all_events])
    methods_present = sorted(set(event["method"] for event in all_events))
    positive_works = int(np.sum(differences > 0))
    gates = {
        "support": bool(len(all_events) >= 20 and len(per_work) >= 4 and methods_present == list(METHODS)),
        "destination_rival": bool(len(percentiles) and np.median(percentiles) >= .90 and np.mean(percentiles >= .95) >= .50),
        "precursor_difference": bool(
            len(differences)
            and differences.mean() > 0
            and bootstrap_ci(differences)[0] > 0
            and positive_works >= int(np.ceil(.6*len(differences)))
        ),
        "high_ambiguity_ratio": bool(event_high/max(control_high, 1e-12) >= 1.5),
    }
    gates["pass"] = bool(all(gates.values()))
    result = {
        "raw_bursts": dict(raw_counts),
        "matched_events": len(all_events),
        "works": len(per_work),
        "methods": methods_present,
        "destination": {
            "median_percentile": float(np.median(percentiles)) if len(percentiles) else None,
            "percentile_95_fraction": float(np.mean(percentiles >= .95)) if len(percentiles) else None,
            "median_similarity": float(np.median([event["destination_similarity"] for event in all_events])) if all_events else None,
        },
        "precursor": {
            "work_equal_mean_maxsim_difference": float(differences.mean()) if len(differences) else None,
            "work_bootstrap95": bootstrap_ci(differences) if len(differences) else None,
            "positive_works": positive_works,
            "required_positive_works": int(np.ceil(.6*len(differences))) if len(differences) else 0,
            "event_high_ambiguity_rate": event_high,
            "control_high_ambiguity_rate": control_high,
            "risk_ratio": event_high/max(control_high, 1e-12),
            "per_work": per_work,
        },
        "outcomes": dict(sorted(Counter(event["outcome"] for event in all_events).items())),
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
