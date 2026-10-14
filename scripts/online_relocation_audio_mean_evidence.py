"""Prospective Vienna validation of parameter-free mean-chroma candidate evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from online_relocation_vienna_audio import (  # noqa: E402
    METHODS, bootstrap_ci, ground_truth, make_events, merged_score, path_valid,
    run_tracker, score_geometry, score_seconds, write_json,
)
from score_following.features import notes_to_chroma_stream  # noqa: E402


DATA = ROOT / "data" / "vienna4x22_rematched"
AUDIO = ROOT / "data" / "vienna4x22_audio_mean_evidence"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
PROTOCOL = ROOT / "AUDIO_MEAN_EVIDENCE_VALIDATION_PROTOCOL.md"
OUT = ROOT / "results" / "online_relocation_audio_mean_evidence"
EXPECTED_COMMIT = "40fd6e7602bd9e596cea1beafdcee4853b9f2c98"
EXPECTED_MODEL = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"
PERFORMERS = ("p02", "p05", "p08", "p11", "p14", "p17", "p20", "p22")
CASES = tuple(
    {
        "case": f"{stem}_{performer}", "work": work, "performer": performer,
        "stem": f"{stem}_{performer}", "score": score,
    }
    for stem, work, score in (
        ("Chopin_op38", "Chopin/Op38", "Chopin_op38.musicxml"),
        ("Chopin_op10_no3", "Chopin/Op10No3", "Chopin_op10_no3.musicxml"),
        ("Mozart_K331_1st-mov", "Mozart/K331_1", "Mozart_K331_1st-mov.musicxml"),
        ("Schubert_D783_no15", "Schubert/D783No15", "Schubert_D783_no15.musicxml"),
    )
    for performer in PERFORMERS
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def augment(event: dict[str, object]) -> dict[str, object]:
    origin_support = float(event["delayed"][14])
    destination_support = float(event["delayed"][15])
    action = "accept" if destination_support >= origin_support else "hold"
    loss = float(event["accept_loss"] if action == "accept" else event["hold_loss"])
    oracle = min(float(event["accept_loss"]), float(event["hold_loss"]))
    return {
        **event,
        "mean_evidence_action": action,
        "mean_evidence_margin": destination_support - origin_support,
        "mean_evidence_regret": loss - oracle,
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "external" / "matchmaker"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT or digest(MODEL_PATH) != EXPECTED_MODEL:
        raise RuntimeError("frozen dependency drift")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    files = [Path(__file__), PROTOCOL, MODEL_PATH, AUDIO / "manifest.json"]
    for spec in CASES:
        files += [
            AUDIO / f"{spec['stem']}.wav",
            DATA / "match" / f"{spec['stem']}.match",
            DATA / "musicxml" / spec["score"],
        ]
    freeze = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in files}
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", freeze)
    write_json(OUT / "state.json", {"status": "RUNNING"})

    processed = []
    implementation = True
    with threadpool_limits(limits=2):
        for number, spec in enumerate(CASES, 1):
            print(f"[{number}/{len(CASES)}] {spec['case']}", flush=True)
            score_path = DATA / "musicxml" / spec["score"]
            audio_path = AUDIO / f"{spec['stem']}.wav"
            score = merged_score(score_path)
            trackers = {}
            tempo = None
            for method in METHODS:
                raw, runtime, current_tempo = run_tracker(score_path, audio_path, method)
                tempo = current_tempo if tempo is None else tempo
                if abs(current_tempo - tempo) > 1e-9:
                    raise RuntimeError("tracker tempo mismatch")
                beat_min, beat_max, _, _ = score_geometry(score, tempo)
                valid = path_valid(raw, beat_min, beat_max)
                implementation = implementation and valid
                raw_origin = float(
                    score.inv_beat_map(beat_min)
                    / score.quarter_duration_map(score.inv_beat_map(beat_min)) * (60 / tempo)
                )
                converted = raw.copy()
                converted[1] = score_seconds(score, raw[1], tempo, raw_origin)
                trackers[method] = {
                    "path_seconds": converted.tolist(), "path_points": int(raw.shape[1]),
                    "runtime_sec": runtime, "tempo": tempo, "path_valid": valid,
                }
                print(f"  {method}: points={raw.shape[1]} valid={valid}", flush=True)
            beat_min, _, score_duration, notes = score_geometry(score, float(tempo))
            raw_origin = float(
                score.inv_beat_map(beat_min)
                / score.quarter_duration_map(score.inv_beat_map(beat_min)) * (60 / float(tempo))
            )
            query, truth, gt_audit = ground_truth(
                DATA / "match" / f"{spec['stem']}.match", score, float(tempo), raw_origin
            )
            score_chroma = notes_to_chroma_stream(notes, score_duration, 0.1)
            case = {
                "case": spec["case"], "work": spec["work"], "performer": spec["performer"],
                "audio": str(audio_path), "query_time": query.tolist(), "truth": truth.tolist(),
                "score_duration": score_duration, "score_chroma": score_chroma.tolist(),
                "gt_audit": gt_audit, "trackers": trackers,
            }
            case["events"] = [augment(event) for event in make_events(case, model)] if implementation else []
            write_json(OUT / f"{spec['case']}.json", case)
            processed.append(case)
            print(f"  anchors={len(query)} events={len(case['events'])}", flush=True)

    events = [event for case in processed for event in case["events"]]
    write_json(OUT / "events.json", events)
    works = sorted({event["work"] for event in events})
    performances = sorted({event["case"] for event in events})
    methods = sorted({event["method"] for event in events})
    per_work = []
    for work in works:
        subset = [event for event in events if event["work"] == work]
        per_work.append({
            "work": work, "events": len(subset),
            "performances": len({event["case"] for event in subset}),
            "global_regret": float(np.mean([event["global_regret"] for event in subset])),
            "frozen_delayed_regret": float(np.mean([event["delayed_regret"] for event in subset])),
            "mean_evidence_regret": float(np.mean([event["mean_evidence_regret"] for event in subset])),
        })
    difference = np.asarray(
        [row["global_regret"] - row["mean_evidence_regret"] for row in per_work], dtype=float
    )
    ci = bootstrap_ci(difference) if len(difference) else [None, None]
    holds = [event for event in events if event["mean_evidence_action"] == "hold"]
    precision = float(np.mean([event["oracle_action"] == "hold" for event in holds])) if holds else None
    gates = {
        "implementation": bool(implementation),
        "support": len(events) >= 20 and len(performances) >= 12 and len(works) >= 3
        and methods == list(METHODS),
        "work_equal_improvement": bool(len(difference) and difference.mean() > 0),
        "bootstrap_lower_positive": bool(len(difference) and ci[0] > 0),
        "no_material_work_worsening": bool(
            len(difference) and np.all(-difference <= 0.05 + 1e-12)
        ),
        "hold_precision": bool(len(holds) >= 5 and precision is not None and precision >= 0.8),
    }
    gates["pass"] = all(gates.values())
    nonzero = difference[np.abs(difference) > 1e-12]
    exact = None
    if len(nonzero):
        means = [
            np.mean(nonzero * np.asarray(signs))
            for signs in itertools.product((-1, 1), repeat=len(nonzero))
        ]
        exact = float(np.mean(np.asarray(means) >= nonzero.mean() - 1e-15))
    result = {
        "status": "PASS" if gates["pass"] else "FAIL",
        "scope": "prospective new-performer validation on the same four Vienna works",
        "events": len(events), "performances": len(performances), "works": len(works),
        "methods": methods, "mean_evidence_holds": len(holds), "hold_precision": precision,
        "oracle_holds": int(sum(event["oracle_action"] == "hold" for event in events)),
        "work_equal_improvement": float(difference.mean()) if len(difference) else None,
        "bootstrap95": ci, "exact_sign_flip_p_one_sided": exact,
        "per_work": per_work, "gates": gates,
    }
    if any(digest(ROOT / relative) != expected for relative, expected in freeze.items()):
        raise RuntimeError("input drift")
    write_json(OUT / "result.json", result)
    write_json(OUT / "state.json", {"status": "COMPLETE", "pass": gates["pass"]})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
