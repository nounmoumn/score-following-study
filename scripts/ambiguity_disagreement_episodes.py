"""Development-only disagreement taxonomy and structural-rival withhold test."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from matchmaker_ambiguity_rescue import fit_ridge, predict_ridge, rolling_state  # noqa: E402
from score_following.features import load_notes, notes_to_chroma_stream  # noqa: E402


SOURCE = ROOT / "results" / "matchmaker_ambiguity_rescue_development"
PROTOCOL = ROOT / "AMBIGUITY_DISAGREEMENT_EPISODE_PROTOCOL.md"
OUT = ROOT / "results" / "ambiguity_disagreement_episodes"
PAIR = ("arzt", "outerhmm")
HOP = 0.1
WINDOW_FRAMES = 20
DISAGREEMENT = 0.5
MIN_SEPARATION = 2.0
SIMILARITY = 0.80
CAP = 5.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def direct_similarity(
    chroma: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    output = np.full(len(left), np.nan, dtype=float)
    for index, (left_second, right_second) in enumerate(zip(left, right)):
        li = int(round(left_second / HOP)) - 1
        ri = int(round(right_second / HOP)) - 1
        if li < 0 or ri < 0 or li + WINDOW_FRAMES > len(chroma) or ri + WINDOW_FRAMES > len(chroma):
            continue
        first = chroma[li : li + WINDOW_FRAMES].reshape(-1)
        second = chroma[ri : ri + WINDOW_FRAMES].reshape(-1)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 0:
            output[index] = float(np.dot(first, second) / denominator)
    return output


def case_rows(case: dict[str, object]) -> dict[str, object]:
    left_tracker, right_tracker = (case["trackers"][method] for method in PAIR)
    time_axis = np.asarray(case["query_time"], float)
    left = np.asarray(left_tracker["position"], float)
    right = np.asarray(right_tracker["position"], float)
    left_update = np.asarray(left_tracker["update_time"], float)
    right_update = np.asarray(right_tracker["update_time"], float)
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(left_update) & np.isfinite(right_update)
    time_axis, left, right = time_axis[valid], left[valid], right[valid]
    left_update, right_update = left_update[valid], right_update[valid]
    left_loss = np.asarray(left_tracker["loss"], float)[valid]
    right_loss = np.asarray(right_tracker["loss"], float)[valid]
    state = np.column_stack(
        [
            left,
            right,
            np.abs(left-right),
            rolling_state(left, left_update, time_axis),
            rolling_state(right, right_update, time_axis),
        ]
    )
    score_path = ROOT / "asap_dataset" / case["score"]
    notes = load_notes(score_path)
    duration = max(note[1] for note in notes)
    chroma = notes_to_chroma_stream(notes, duration, HOP)
    similarity = direct_similarity(chroma, left, right)
    separation = np.abs(left-right)
    disagreement = separation > DISAGREEMENT
    structural = (
        disagreement
        & (separation >= MIN_SEPARATION)
        & np.isfinite(similarity)
        & (similarity >= SIMILARITY)
    )
    return {
        "case": case["case"],
        "work": case["work"],
        "time": time_axis,
        "left": left,
        "right": right,
        "left_loss": left_loss,
        "right_loss": right_loss,
        "target": left_loss-right_loss,
        "state": state,
        "similarity": similarity,
        "separation": separation,
        "disagreement": disagreement,
        "structural": structural,
    }


def add_selector_predictions(rows: list[dict[str, object]]) -> None:
    for held_out in rows:
        training = [row for row in rows if row["work"] != held_out["work"]]
        train_state = np.row_stack([row["state"] for row in training])
        train_target = np.concatenate([row["target"] for row in training])
        groups = np.concatenate(
            [np.repeat(row["work"], len(row["target"])) for row in training]
        )
        model = fit_ridge(train_state, train_target, groups)
        prediction = predict_ridge(held_out["state"], model)
        choose_right = prediction >= 0
        selected = np.where(choose_right, held_out["right_loss"], held_out["left_loss"])
        oracle = np.minimum(held_out["left_loss"], held_out["right_loss"])
        held_out["prediction"] = prediction
        held_out["selected_loss"] = selected
        held_out["selector_error"] = selected > 0.5
        held_out["decision_regret"] = selected-oracle
        held_out["high_regret"] = selected-oracle >= 0.5


def episode_groups(mask: np.ndarray, time_axis: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    groups = [[int(indices[0])]]
    for index in indices[1:]:
        if time_axis[index] - time_axis[groups[-1][-1]] <= 2.0:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [
        np.asarray(group, dtype=int)
        for group in groups
        if len(group) >= 2 and time_axis[group[-1]]-time_axis[group[0]] >= 0.5
    ]


def finite_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def classify_episode(row: dict[str, object], indices: np.ndarray, number: int) -> dict[str, object]:
    time_axis = row["time"][indices]
    left, right = row["left"][indices], row["right"][indices]
    similarity = row["similarity"][indices]
    separation = row["separation"][indices]
    left_jump = np.abs(np.diff(left))
    right_jump = np.abs(np.diff(right))
    dt = np.diff(time_axis)
    speed_difference = np.abs(np.diff(left)/dt - np.diff(right)/dt)
    similarity_median = finite_median(similarity)
    separation_median = float(np.median(separation))
    max_jump = float(max(np.max(left_jump, initial=0), np.max(right_jump, initial=0)))
    speed_median = float(np.median(speed_difference)) if len(speed_difference) else 0.0
    if time_axis[0] <= 10.0:
        category = "initialization"
    elif max_jump >= 2.0:
        category = "relocation"
    elif similarity_median is not None and similarity_median >= SIMILARITY and separation_median >= MIN_SEPARATION:
        category = "structural_rival"
    elif speed_median >= 0.50:
        category = "tempo_divergence"
    else:
        category = "residual"
    target = row["target"][indices]
    nonzero = np.sign(target[np.abs(target) > 1e-12])
    winner_switches = int(np.sum(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
    return {
        "episode_id": f"{row['case']}_e{number:03d}",
        "case": row["case"],
        "work": row["work"],
        "category": category,
        "start_sec": float(time_axis[0]),
        "end_sec": float(time_axis[-1]),
        "duration_sec": float(time_axis[-1]-time_axis[0]),
        "beats": int(len(indices)),
        "median_direct_similarity": similarity_median,
        "median_score_separation": separation_median,
        "max_tracker_jump": max_jump,
        "median_speed_difference": speed_median,
        "winner_switches": winner_switches,
        "selector_error_rate": float(np.mean(row["selector_error"][indices])),
        "high_regret_rate": float(np.mean(row["high_regret"][indices])),
        "mean_decision_regret": float(np.mean(row["decision_regret"][indices])),
        "mean_arzt_loss": float(np.mean(row["left_loss"][indices])),
        "mean_outerhmm_loss": float(np.mean(row["right_loss"][indices])),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    saved_state = json.loads((SOURCE / "state.json").read_text(encoding="utf-8"))
    if saved_state.get("selected_pair") != list(PAIR):
        raise ValueError("Pinned pair does not match completed development result")
    source_files = [SOURCE / f"fd{i:03d}.json" for i in range(8)]
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in source_files]
    score_files = [ROOT / "asap_dataset" / case["score"] for case in cases]
    frozen_files = [Path(__file__), PROTOCOL, SOURCE / "result.json", SOURCE / "state.json", *source_files, *score_files]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in frozen_files}
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", freeze)
    write_json(OUT / "state.json", {"status": "RUNNING", "lockbox_authorized": False})

    rows = [case_rows(case) for case in cases]
    add_selector_predictions(rows)
    episodes = []
    for row in rows:
        for number, indices in enumerate(episode_groups(row["disagreement"], row["time"]), 1):
            episodes.append(classify_episode(row, indices, number))
    write_json(OUT / "episodes.json", episodes)

    work_records = []
    flagged_beats = 0
    flagged_works = 0
    for row in rows:
        disagreement = row["disagreement"]
        flagged = row["structural"]
        unflagged = disagreement & ~flagged
        flagged_count, unflagged_count = int(flagged.sum()), int(unflagged.sum())
        flagged_beats += flagged_count
        flagged_works += int(flagged_count > 0)
        record = {
            "work": row["work"],
            "disagreement_beats": int(disagreement.sum()),
            "flagged_beats": flagged_count,
            "unflagged_beats": unflagged_count,
            "coverage": float(flagged_count/max(int(disagreement.sum()), 1)),
        }
        if flagged_count and unflagged_count:
            record.update(
                {
                    "selector_error_flagged": float(np.mean(row["selector_error"][flagged])),
                    "selector_error_unflagged": float(np.mean(row["selector_error"][unflagged])),
                    "high_regret_flagged": float(np.mean(row["high_regret"][flagged])),
                    "high_regret_unflagged": float(np.mean(row["high_regret"][unflagged])),
                }
            )
            record["selector_error_lift"] = record["selector_error_flagged"]-record["selector_error_unflagged"]
            record["high_regret_lift"] = record["high_regret_flagged"]-record["high_regret_unflagged"]
        work_records.append(record)
    comparable = [record for record in work_records if "selector_error_flagged" in record]
    coverage = float(np.mean([record["coverage"] for record in work_records]))
    error_flagged = float(np.mean([record["selector_error_flagged"] for record in comparable])) if comparable else 0.0
    error_unflagged = float(np.mean([record["selector_error_unflagged"] for record in comparable])) if comparable else 0.0
    regret_flagged = float(np.mean([record["high_regret_flagged"] for record in comparable])) if comparable else 0.0
    regret_unflagged = float(np.mean([record["high_regret_unflagged"] for record in comparable])) if comparable else 0.0
    error_ratio = error_flagged/max(error_unflagged, 1e-12)
    regret_ratio = regret_flagged/max(regret_unflagged, 1e-12)
    positive_works = int(sum(record["selector_error_lift"] > 0 for record in comparable))
    required_positive = max(3, int(np.ceil(0.6*len(comparable))))
    gates = {
        "support": bool(flagged_beats >= 20 and flagged_works >= 3 and coverage >= 0.05),
        "selector_error_ratio": bool(error_ratio >= 1.5),
        "high_regret_ratio": bool(regret_ratio >= 1.5),
        "positive_work_fraction": bool(len(comparable) >= 3 and positive_works >= required_positive),
    }
    gates["pass"] = bool(all(gates.values()))
    result = {
        "pair": list(PAIR),
        "cases": len(rows),
        "episodes": len(episodes),
        "episode_categories": dict(sorted(Counter(episode["category"] for episode in episodes).items())),
        "detector": {
            "flagged_beats": flagged_beats,
            "flagged_works": flagged_works,
            "work_equal_disagreement_coverage": coverage,
            "comparable_works": len(comparable),
            "selector_error": {"flagged": error_flagged, "unflagged": error_unflagged, "risk_ratio": error_ratio},
            "high_regret": {"flagged": regret_flagged, "unflagged": regret_unflagged, "risk_ratio": regret_ratio},
            "positive_selector_error_lift_works": positive_works,
            "required_positive_works": required_positive,
            "per_work": work_records,
        },
        "gates": gates,
        "lockbox_authorized": bool(gates["pass"]),
    }
    if any(sha256(ROOT / relative) != digest for relative, digest in freeze.items()):
        raise RuntimeError("Frozen input changed during episode analysis")
    write_json(OUT / "result.json", result)
    write_json(OUT / "state.json", {"status": "COMPLETE", "lockbox_authorized": bool(gates["pass"])})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
