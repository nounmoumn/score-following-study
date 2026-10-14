"""Exploratory ordered-continuation evidence on the five opened Vienna events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_following.features import audio_chroma  # noqa: E402


IN_DIR = ROOT / "results" / "online_relocation_vienna_audio"
OUT_DIR = ROOT / "results" / "audio_relocation_continuation_diagnostic"
HOP = 0.1
DELAY = 2.0
WINDOW = int(round(DELAY / HOP))


def frame_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def diagonal_similarity(observed: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(frame_cosine(observed, candidate)))


def dtw_cost(observed: np.ndarray, candidate: np.ndarray) -> float:
    """Normalized cosine DTW cost with the three standard monotone moves."""
    local = 1.0 - np.asarray([
        frame_cosine(np.repeat(row[None, :], len(candidate), axis=0), candidate)
        for row in observed
    ])
    rows, columns = local.shape
    cost = np.full((rows + 1, columns + 1), np.inf)
    length = np.zeros((rows + 1, columns + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            options = ((cost[i - 1, j], length[i - 1, j]),
                       (cost[i, j - 1], length[i, j - 1]),
                       (cost[i - 1, j - 1], length[i - 1, j - 1]))
            best_cost, best_length = min(options, key=lambda item: item[0])
            cost[i, j] = best_cost + local[i - 1, j - 1]
            length[i, j] = best_length + 1
    return float(cost[rows, columns] / length[rows, columns])


def candidate_window(score_chroma: np.ndarray, position: float) -> np.ndarray | None:
    start = int(np.floor(position / HOP))
    stop = start + WINDOW
    if start < 0 or stop > len(score_chroma):
        return None
    return score_chroma[start:stop]


def action_regret(event: dict[str, object], action: str) -> float:
    loss = float(event["accept_loss"] if action == "accept" else event["hold_loss"])
    return loss - min(float(event["accept_loss"]), float(event["hold_loss"]))


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"Refuse overwrite: {OUT_DIR}")
    events = json.loads((IN_DIR / "events.json").read_text(encoding="utf-8"))
    cases = {}
    output = []
    for event in events:
        case_id = str(event["case"])
        if case_id not in cases:
            case = json.loads((IN_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
            chroma, _ = audio_chroma(Path(case["audio"]), hop=HOP)
            cases[case_id] = (case, np.asarray(chroma, dtype=float))
        case, performance_chroma = cases[case_id]
        score_chroma = np.asarray(case["score_chroma"], dtype=float)
        start = int(np.floor(float(event["t0"]) / HOP))
        observed = performance_chroma[start:start + WINDOW]
        origin = candidate_window(score_chroma, float(event["origin"]))
        destination = candidate_window(score_chroma, float(event["destination"]))
        if len(observed) != WINDOW or origin is None or destination is None:
            raise AssertionError(f"invalid candidate window: {event['event_id']}")

        origin_diagonal = diagonal_similarity(observed, origin)
        destination_diagonal = diagonal_similarity(observed, destination)
        origin_dtw = dtw_cost(observed, origin)
        destination_dtw = dtw_cost(observed, destination)
        diagonal_action = "accept" if destination_diagonal >= origin_diagonal else "hold"
        dtw_action = "accept" if destination_dtw <= origin_dtw else "hold"
        mean_action = "accept" if float(event["delayed"][15]) >= float(event["delayed"][14]) else "hold"
        output.append({
            "event_id": event["event_id"],
            "work": event["work"],
            "oracle_action": event["oracle_action"],
            "mean_chroma_action": mean_action,
            "ordered_diagonal_action": diagonal_action,
            "local_dtw_action": dtw_action,
            "origin_mean_similarity": float(event["delayed"][14]),
            "destination_mean_similarity": float(event["delayed"][15]),
            "origin_diagonal_similarity": origin_diagonal,
            "destination_diagonal_similarity": destination_diagonal,
            "origin_dtw_cost": origin_dtw,
            "destination_dtw_cost": destination_dtw,
            "mean_chroma_regret": action_regret(event, mean_action),
            "ordered_diagonal_regret": action_regret(event, diagonal_action),
            "local_dtw_regret": action_regret(event, dtw_action),
        })

    methods = {
        "mean_chroma": ("mean_chroma_action", "mean_chroma_regret"),
        "ordered_diagonal": ("ordered_diagonal_action", "ordered_diagonal_regret"),
        "local_dtw": ("local_dtw_action", "local_dtw_regret"),
    }
    summary = {}
    for name, (action_key, regret_key) in methods.items():
        summary[name] = {
            "correct": int(sum(row[action_key] == row["oracle_action"] for row in output)),
            "events": len(output),
            "holds": int(sum(row[action_key] == "hold" for row in output)),
            "oracle_hold_true_positives": int(sum(
                row[action_key] == "hold" and row["oracle_action"] == "hold" for row in output
            )),
            "mean_regret": float(np.mean([row[regret_key] for row in output])),
        }
    result = {
        "status": "EXPLORATORY_ONLY",
        "reason": "all five Vienna event outcomes were already open before this diagnostic",
        "definition": {
            "delay_sec": DELAY,
            "hop_sec": HOP,
            "frames": WINDOW,
            "diagonal": "mean framewise cosine similarity",
            "dtw": "normalized cosine cost, monotone horizontal/vertical/diagonal moves",
        },
        "summary": summary,
        "events": output,
    }
    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
