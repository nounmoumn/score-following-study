"""Frozen relocation-validator shadow transfer on official Matchmaker audio examples."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import partitura as pt
import soundfile as sf
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ambiguity_induced_relocation import burst_events, frame_index  # noqa: E402
from matchmaker_acquisition_smoke import score_clock  # noqa: E402
from score_following.features import audio_chroma, load_notes, notes_to_chroma_stream  # noqa: E402


ASSETS = ROOT / "external" / "matchmaker" / "matchmaker" / "assets"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
PROTOCOL = ROOT / "ONLINE_RELOCATION_AUDIO_SHADOW_PROTOCOL.md"
OUT = ROOT / "results" / "online_relocation_audio_shadow"
EXPECTED_COMMIT = "40fd6e7602bd9e596cea1beafdcee4853b9f2c98"
EXPECTED_MODEL = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"
METHODS = ("arzt", "outerhmm")
DELAY = 2.0
CAP = 5.0
HOP = 0.1
WINDOW = 20
CASES = (
    {
        "case": "audio_bach858",
        "work": "Bach/Fugue_BWV858",
        "score": "Bach-fugue_bwv_858.mid",
        "audio": "Bach-fugue_bwv_858.mp3",
        "annotation": "Bach-fugue_bwv_858_beat_annotations.txt",
        "prefix_sec": 30.0,
    },
    {
        "case": "audio_mozart_k265_v1",
        "work": "Mozart/K265_Variation1",
        "score": "mozart_k265_var1.mid",
        "audio": "mozart_k265_var1.mp3",
        "annotation": "mozart_k265_var1_annotations.txt",
        "prefix_sec": 12.0,
    },
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def predict(features: list[float], model: dict[str, list[float]]) -> float:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    return float(coefficient[0] + ((x - mean) / scale) @ coefficient[1:])


def run_audio_tracker(score: Path, audio: Path, method: str) -> tuple[np.ndarray, float]:
    from matchmaker import Matchmaker

    started = time.perf_counter()
    tracker = Matchmaker(
        score_file=score,
        performance_file=audio,
        input_type="audio",
        method=method,
        wait=False,
        unfold_score=False,
    )
    list(tracker.run(verbose=False))
    path = np.asarray(tracker.score_follower.alignment_path, dtype=float)
    return path, float(time.perf_counter() - started)


def latest(path: np.ndarray, time_point: float) -> float | None:
    index = int(np.searchsorted(path[0], time_point, side="right") - 1)
    return float(path[1, index]) if index >= 0 else None


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator > 0 else 0.0


def support(
    performance_chroma: np.ndarray,
    score_chroma: np.ndarray,
    t0: float,
    origin: float,
    destination: float,
) -> tuple[float, float] | None:
    start = max(0, int(np.floor(t0 / HOP)))
    end = min(len(performance_chroma), int(np.ceil((t0 + DELAY) / HOP)))
    origin_index = frame_index(origin, len(score_chroma) - WINDOW + 1)
    destination_index = frame_index(destination, len(score_chroma) - WINDOW + 1)
    if end - start < WINDOW - 1 or origin_index is None or destination_index is None:
        return None
    observed = performance_chroma[start:end].mean(axis=0)
    return (
        cosine(observed, score_chroma[origin_index : origin_index + WINDOW].mean(axis=0)),
        cosine(observed, score_chroma[destination_index : destination_index + WINDOW].mean(axis=0)),
    )


def validate_path(path: np.ndarray, beat_min: float, beat_max: float) -> bool:
    return bool(
        path.ndim == 2
        and path.shape[0] == 2
        and path.shape[1] >= 20
        and np.isfinite(path).all()
        and np.all(np.diff(path[0]) >= -1e-12)
        and np.all(path[1] >= beat_min - 1e-7)
        and np.all(path[1] <= beat_max + 1e-7)
    )


def prefix_audit(full: np.ndarray, prefix: np.ndarray, cutoff: float) -> dict[str, object]:
    safe = cutoff - 0.25
    left = full[:, full[0] <= safe]
    right = prefix[:, prefix[0] <= safe]
    same_shape = left.shape == right.shape
    residual = float(np.max(np.abs(left - right), initial=0.0)) if same_shape else None
    return {
        "safe_cutoff_sec": safe,
        "full_points": int(left.shape[1]),
        "prefix_points": int(right.shape[1]),
        "same_shape": same_shape,
        "max_abs_residual": residual,
        "pass": bool(same_shape and residual is not None and residual <= 1e-9),
    }


def make_events(case: dict[str, object], model: dict[str, object]) -> list[dict[str, object]]:
    score_notes = load_notes(Path(case["score_path"]))
    score_duration = max(note[1] for note in score_notes)
    score_chroma = notes_to_chroma_stream(score_notes, score_duration, HOP)
    performance_chroma, performance_duration = audio_chroma(Path(case["audio_wav"]), hop=HOP)
    truth_time = np.asarray(case["query_time"], dtype=float)
    truth_position = np.asarray(case["truth"], dtype=float)
    paths = {method: np.asarray(case["trackers"][method]["path_seconds"], dtype=float) for method in METHODS}
    events: list[dict[str, object]] = []
    for method in METHODS:
        other = METHODS[1] if method == METHODS[0] else METHODS[0]
        path, other_path = paths[method], paths[other]
        for event_index in burst_events(path):
            t0 = float(path[0, event_index])
            t1 = t0 + DELAY
            if t1 > truth_time[-1] or t1 > path[0, -1] or t1 > other_path[0, -1] or t1 > performance_duration:
                continue
            origin = float(path[1, event_index - 1])
            destination = float(path[1, event_index])
            relocated_t1 = latest(path, t1)
            other_t0, other_t1 = latest(other_path, t0), latest(other_path, t1)
            if relocated_t1 is None or other_t0 is None or other_t1 is None:
                continue
            evidence = support(performance_chroma, score_chroma, t0, origin, destination)
            if evidence is None:
                continue
            origin_support, destination_support = evidence
            truth_t1 = float(np.interp(t1, truth_time, truth_position))
            hold_loss = min(abs(origin - truth_t1), CAP)
            accept_loss = min(abs(relocated_t1 - truth_t1), CAP)
            immediate = [
                float(np.log1p(abs(destination - origin))),
                float(np.sign(destination - origin)),
                origin / max(score_duration, 1e-12),
                destination / max(score_duration, 1e-12),
                min(abs(other_t0 - origin), 10.0),
                min(abs(other_t0 - destination), 10.0),
                float(np.clip(abs(other_t0 - origin) - abs(other_t0 - destination), -10, 10)),
            ]
            delayed = immediate + [
                min(abs(relocated_t1 - origin), 10.0),
                min(abs(relocated_t1 - destination), 10.0),
                min(abs(other_t1 - origin), 10.0),
                min(abs(other_t1 - destination), 10.0),
                min(abs(relocated_t1 - other_t1), 10.0),
                float(np.clip(abs(other_t1 - origin) - abs(other_t1 - destination), -10, 10)),
                min(abs(relocated_t1 - destination), 10.0),
                origin_support,
                destination_support,
                destination_support - origin_support,
            ]
            pi = predict(immediate, model["immediate"])
            pd = predict(delayed, model["delayed"])
            oracle = min(hold_loss, accept_loss)
            events.append({
                "event_id": f"{case['case']}_{method}_v{len(events)+1:03d}",
                "case": case["case"], "work": case["work"], "method": method,
                "t0": t0, "t1": t1, "origin": origin, "destination": destination,
                "relocated_t1": relocated_t1, "other_t0": other_t0, "other_t1": other_t1,
                "truth_t1": truth_t1, "hold_loss": hold_loss, "accept_loss": accept_loss,
                "oracle_action": "accept" if accept_loss <= hold_loss else "hold",
                "immediate": immediate, "delayed": delayed,
                "immediate_prediction": pi, "delayed_prediction": pd,
                "global_regret": accept_loss - oracle,
                "immediate_regret": (accept_loss if pi >= 0 else hold_loss) - oracle,
                "delayed_regret": (accept_loss if pd >= 0 else hold_loss) - oracle,
            })
    return events


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "matchmaker"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected Matchmaker commit {commit}")
    if digest(MODEL_PATH) != EXPECTED_MODEL:
        raise RuntimeError("frozen model drift")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    input_paths = [Path(__file__), PROTOCOL, MODEL_PATH]
    input_paths += [ASSETS / case[key] for case in CASES for key in ("score", "audio", "annotation")]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in input_paths}
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", freeze)
    write_json(OUT / "state.json", {"status": "RUNNING"})
    processed: list[dict[str, object]] = []
    implementation_ok = True
    with threadpool_limits(limits=2):
        for index, spec in enumerate(CASES, 1):
            print(f"[{index}/{len(CASES)}] {spec['case']}", flush=True)
            score = ASSETS / spec["score"]
            source_audio = ASSETS / spec["audio"]
            annotation = np.loadtxt(ASSETS / spec["annotation"], dtype=float)
            if annotation.ndim != 2 or annotation.shape[1] < 2 or len(annotation) < 20:
                raise RuntimeError("invalid annotation")
            query, truth = annotation[:, 0], annotation[:, 1]
            order = np.argsort(query)
            query, truth = query[order], truth[order]
            if not np.isfinite(query).all() or not np.isfinite(truth).all() or np.any(np.diff(query) < 0):
                raise RuntimeError("invalid annotation clock")
            audio, sample_rate = librosa.load(source_audio, sr=None, mono=True)
            full_wav = OUT / f"{spec['case']}_decoded.wav"
            prefix_wav = OUT / f"{spec['case']}_prefix.wav"
            sf.write(full_wav, audio, sample_rate, subtype="FLOAT")
            sf.write(prefix_wav, audio[: int(round(float(spec['prefix_sec']) * sample_rate))], sample_rate, subtype="FLOAT")
            beats, seconds, clock_audit = score_clock(score)
            score_array = pt.load_score(str(score)).note_array()
            beat_end = float(np.max(
                np.asarray(score_array["onset_beat"], float)
                + np.asarray(score_array["duration_beat"], float)
            ))
            score_note_seconds = load_notes(score)
            second_end = float(max(note[1] for note in score_note_seconds))
            if beat_end > beats[-1] + 1e-12:
                beats = np.append(beats, beat_end)
                seconds = np.append(seconds, second_end)
            clock_audit["domain_beat_range"] = [0.0, beat_end]
            clock_audit["domain_second_end"] = second_end
            trackers: dict[str, object] = {}
            for method in METHODS:
                full_path, runtime = run_audio_tracker(score, full_wav, method)
                prefix_path, prefix_runtime = run_audio_tracker(score, prefix_wav, method)
                # Score time begins at beat 0 even when the first note onset is later.
                valid = validate_path(full_path, 0.0, beat_end)
                causal = prefix_audit(full_path, prefix_path, float(spec["prefix_sec"]))
                implementation_ok = implementation_ok and valid and bool(causal["pass"])
                converted = full_path.copy()
                converted[1] = np.interp(full_path[1], beats, seconds)
                trackers[method] = {
                    "path_seconds": converted.tolist(),
                    "path_points": int(full_path.shape[1]),
                    "runtime_sec": runtime,
                    "prefix_runtime_sec": prefix_runtime,
                    "path_valid": valid,
                    "prefix_causality": causal,
                }
                print(f"  {method}: points={full_path.shape[1]} valid={valid} causal={causal['pass']}", flush=True)
            case = {
                "case": spec["case"], "work": spec["work"],
                "score_path": str(score), "audio_wav": str(full_wav),
                "query_time": query.tolist(), "truth": truth.tolist(),
                "trackers": trackers, "score_clock_audit": clock_audit,
            }
            events = make_events(case, model) if implementation_ok else []
            case["events"] = events
            write_json(OUT / f"{spec['case']}.json", case)
            processed.append(case)
            print(f"  eligible_events={len(events)}", flush=True)
    events = [event for case in processed for event in case["events"]]
    write_json(OUT / "events.json", events)
    works = sorted({event["work"] for event in events})
    methods = sorted({event["method"] for event in events})
    per_work = []
    for work in works:
        subset = [event for event in events if event["work"] == work]
        per_work.append({
            "work": work,
            "events": len(subset),
            "global_regret": float(np.mean([event["global_regret"] for event in subset])),
            "immediate_regret": float(np.mean([event["immediate_regret"] for event in subset])),
            "delayed_regret": float(np.mean([event["delayed_regret"] for event in subset])),
        })
    support_gate = len(events) >= 10 and len(works) == 2 and methods == list(METHODS)
    improvement_gate = bool(per_work) and float(np.mean([
        row["global_regret"] - row["delayed_regret"] for row in per_work
    ])) > 0
    noninferiority_gate = bool(per_work) and all(
        row["delayed_regret"] - row["global_regret"] <= 0.05 + 1e-12 for row in per_work
    )
    holds = [event for event in events if event["delayed_prediction"] < 0]
    hold_precision = (
        float(np.mean([event["oracle_action"] == "hold" for event in holds])) if holds else None
    )
    hold_gate = len(holds) >= 3 and hold_precision is not None and hold_precision >= 0.8
    gates = {
        "implementation": bool(implementation_ok),
        "support": support_gate,
        "work_equal_improvement": improvement_gate,
        "no_material_work_worsening": noninferiority_gate,
        "hold_precision": hold_gate,
    }
    gates["pass"] = all(gates.values())
    result = {
        "status": "PASS" if gates["pass"] else "FAIL",
        "scope": "two-work official audio shadow transfer pilot, not external confirmation",
        "events": len(events), "works": len(works), "methods": methods,
        "holds": len(holds), "hold_precision": hold_precision,
        "per_work": per_work, "gates": gates,
        "runtime_sec": {
            case["case"]: {
                method: case["trackers"][method]["runtime_sec"] for method in METHODS
            } for case in processed
        },
    }
    if any(digest(ROOT / relative) != expected for relative, expected in freeze.items()):
        raise RuntimeError("input drift")
    write_json(OUT / "result.json", result)
    write_json(OUT / "state.json", {"status": "COMPLETE", "pass": gates["pass"]})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
