"""Frozen audio relocation-validator transfer on an outcome-blind Vienna subset."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import partitura as pt
import soundfile as sf
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ambiguity_induced_relocation import burst_events  # noqa: E402
from online_relocation_audio_shadow import (  # noqa: E402
    CAP, DELAY, HOP, METHODS, WINDOW, cosine, latest, predict,
)
from score_following.features import audio_chroma, notes_to_chroma_stream  # noqa: E402


DATA = ROOT / "data" / "vienna4x22_rematched"
AUDIO = ROOT / "data" / "vienna4x22_audio_shadow"
MODEL_PATH = ROOT / "output" / "online_relocation_validator_frozen" / "model.json"
PROTOCOL = ROOT / "ONLINE_RELOCATION_VIENNA_AUDIO_PROTOCOL.md"
OUT = ROOT / "results" / "online_relocation_vienna_audio"
EXPECTED_COMMIT = "40fd6e7602bd9e596cea1beafdcee4853b9f2c98"
EXPECTED_MODEL = "233e85f04b9a50c2805b4c56df5f0f0d7eb95713318b887f1688dae8aea422ff"
SEED = 20260924
CASES = tuple(
    {
        "case": f"{stem}_{performer}",
        "work": work,
        "performer": performer,
        "stem": f"{stem}_{performer}",
        "score": score,
        "prefix_audit": performer == "p01",
    }
    for stem, work, score in (
        ("Chopin_op38", "Chopin/Op38", "Chopin_op38.musicxml"),
        ("Chopin_op10_no3", "Chopin/Op10No3", "Chopin_op10_no3.musicxml"),
        ("Mozart_K331_1st-mov", "Mozart/K331_1", "Mozart_K331_1st-mov.musicxml"),
        ("Schubert_D783_no15", "Schubert/D783No15", "Schubert_D783_no15.musicxml"),
    )
    for performer in ("p01", "p12")
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def merged_score(score_path: Path):
    score = pt.load_musicxml(str(score_path), ignore_invisible_objects=True)
    return pt.score.merge_parts(score.parts)


def run_tracker(score_path: Path, audio_path: Path, method: str) -> tuple[np.ndarray, float, float]:
    from matchmaker import Matchmaker

    started = time.perf_counter()
    tracker = Matchmaker(
        score_file=score_path,
        performance_file=audio_path,
        input_type="audio",
        method=method,
        wait=False,
        unfold_score=False,
    )
    list(tracker.run(verbose=False))
    return (
        np.asarray(tracker.score_follower.alignment_path, dtype=float),
        float(time.perf_counter() - started),
        float(tracker.tempo),
    )


def score_seconds(score, beats: np.ndarray, tempo: float, origin: float) -> np.ndarray:
    values = []
    for beat in np.asarray(beats, dtype=float):
        division = score.inv_beat_map(float(beat))
        quarter = score.quarter_duration_map(division)
        values.append(float(division / quarter * (60.0 / tempo) - origin))
    return np.asarray(values, dtype=float)


def score_geometry(score, tempo: float) -> tuple[float, float, float, list[tuple[float, float, int]]]:
    array = score.note_array()
    onset = np.asarray(array["onset_beat"], dtype=float)
    end = onset + np.asarray(array["duration_beat"], dtype=float)
    beat_min = float(score.beat_map(score.first_point.t))
    beat_max = float(score.beat_map(score.last_point.t))
    raw_origin = float(
        score.inv_beat_map(beat_min)
        / score.quarter_duration_map(score.inv_beat_map(beat_min))
        * (60.0 / tempo)
    )
    starts = score_seconds(score, onset, tempo, raw_origin)
    ends = score_seconds(score, end, tempo, raw_origin)
    notes = [(float(start), float(stop), int(pitch)) for start, stop, pitch in zip(starts, ends, array["pitch"])]
    duration = float(score_seconds(score, np.asarray([beat_max]), tempo, raw_origin)[0])
    return beat_min, beat_max, duration, notes


def ground_truth(match_path: Path, score, tempo: float, origin: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    performance, alignment, match_score = pt.load_match(str(match_path), create_score=True, quiet=True)
    p_array = performance.note_array()
    s_array = match_score.note_array()
    perf = {str(row["id"]): float(row["onset_sec"]) for row in p_array}
    score_beat = {str(row["id"]): float(row["onset_beat"]) for row in s_array}
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in alignment:
        if row.get("label") != "match":
            continue
        sid, pid = str(row["score_id"]), str(row["performance_id"])
        if sid in score_beat and pid in perf:
            grouped[score_beat[sid]].append(perf[pid])
    beats = np.asarray(sorted(grouped), dtype=float)
    query = np.asarray([np.median(grouped[beat]) for beat in beats], dtype=float)
    truth = score_seconds(score, beats, tempo, origin)
    keep = np.r_[True, (np.diff(query) > 1e-9) & (np.diff(truth) > 1e-9)]
    query, truth, beats = query[keep], truth[keep], beats[keep]
    audit = {
        "matched_score_onsets": len(grouped),
        "strict_monotone_anchors": int(len(query)),
        "dropped_nonmonotone_or_duplicate": int(len(grouped) - len(query)),
        "performance_range_sec": [float(query[0]), float(query[-1])],
        "score_position_range_sec": [float(truth[0]), float(truth[-1])],
    }
    if len(query) < 20 or not np.isfinite(query).all() or not np.isfinite(truth).all():
        raise RuntimeError(f"invalid GT anchors in {match_path}")
    if np.any(np.diff(query) <= 0) or np.any(np.diff(truth) <= 0):
        raise RuntimeError(f"nonmonotone GT anchors in {match_path}")
    return query, truth, audit


def path_valid(path: np.ndarray, beat_min: float, beat_max: float) -> bool:
    return bool(
        path.ndim == 2 and path.shape[0] == 2 and path.shape[1] >= 20
        and np.isfinite(path).all() and np.all(np.diff(path[0]) >= -1e-12)
        and np.all(path[1] >= beat_min - 1e-7) and np.all(path[1] <= beat_max + 1e-7)
    )


def prefix_audit(full: np.ndarray, prefix: np.ndarray) -> dict[str, object]:
    safe = 19.75
    left, right = full[:, full[0] <= safe], prefix[:, prefix[0] <= safe]
    same = left.shape == right.shape
    residual = float(np.max(np.abs(left - right), initial=0.0)) if same else None
    return {"same_shape": same, "full_points": int(left.shape[1]), "prefix_points": int(right.shape[1]),
            "max_abs_residual": residual, "pass": bool(same and residual is not None and residual <= 1e-9)}


def make_events(case: dict[str, object], model: dict[str, object]) -> list[dict[str, object]]:
    score_chroma = np.asarray(case["score_chroma"], dtype=float)
    performance_chroma, performance_duration = audio_chroma(Path(case["audio"]), hop=HOP)
    truth_time = np.asarray(case["query_time"], dtype=float)
    truth_position = np.asarray(case["truth"], dtype=float)
    paths = {m: np.asarray(case["trackers"][m]["path_seconds"], dtype=float) for m in METHODS}
    score_duration = float(case["score_duration"])
    events: list[dict[str, object]] = []
    for method in METHODS:
        other = METHODS[1] if method == METHODS[0] else METHODS[0]
        path, other_path = paths[method], paths[other]
        for event_index in burst_events(path):
            t0, t1 = float(path[0, event_index]), float(path[0, event_index] + DELAY)
            if t1 > truth_time[-1] or t1 > path[0, -1] or t1 > other_path[0, -1] or t1 > performance_duration:
                continue
            origin, destination = float(path[1, event_index - 1]), float(path[1, event_index])
            relocated_t1, other_t0, other_t1 = latest(path, t1), latest(other_path, t0), latest(other_path, t1)
            if relocated_t1 is None or other_t0 is None or other_t1 is None:
                continue
            start, end = max(0, int(np.floor(t0 / HOP))), min(len(performance_chroma), int(np.ceil(t1 / HOP)))
            oi, di = int(np.floor(origin / HOP)), int(np.floor(destination / HOP))
            if end - start < WINDOW - 1 or oi < 0 or di < 0 or oi + WINDOW > len(score_chroma) or di + WINDOW > len(score_chroma):
                continue
            observed = performance_chroma[start:end].mean(axis=0)
            origin_support = cosine(observed, score_chroma[oi:oi + WINDOW].mean(axis=0))
            destination_support = cosine(observed, score_chroma[di:di + WINDOW].mean(axis=0))
            truth_t1 = float(np.interp(t1, truth_time, truth_position))
            hold_loss, accept_loss = min(abs(origin - truth_t1), CAP), min(abs(relocated_t1 - truth_t1), CAP)
            immediate = [float(np.log1p(abs(destination-origin))), float(np.sign(destination-origin)),
                origin/max(score_duration,1e-12), destination/max(score_duration,1e-12),
                min(abs(other_t0-origin),10.0), min(abs(other_t0-destination),10.0),
                float(np.clip(abs(other_t0-origin)-abs(other_t0-destination),-10,10))]
            delayed = immediate + [min(abs(relocated_t1-origin),10.0), min(abs(relocated_t1-destination),10.0),
                min(abs(other_t1-origin),10.0), min(abs(other_t1-destination),10.0), min(abs(relocated_t1-other_t1),10.0),
                float(np.clip(abs(other_t1-origin)-abs(other_t1-destination),-10,10)), min(abs(relocated_t1-destination),10.0),
                origin_support, destination_support, destination_support-origin_support]
            pi, pd = predict(immediate, model["immediate"]), predict(delayed, model["delayed"])
            oracle = min(hold_loss, accept_loss)
            events.append({"event_id":f"{case['case']}_{method}_v{len(events)+1:03d}","case":case["case"],"work":case["work"],
                "performer":case["performer"],"method":method,"t0":t0,"t1":t1,"origin":origin,"destination":destination,
                "relocated_t1":relocated_t1,"other_t0":other_t0,"other_t1":other_t1,"truth_t1":truth_t1,
                "hold_loss":hold_loss,"accept_loss":accept_loss,"oracle_action":"accept" if accept_loss<=hold_loss else "hold",
                "immediate":immediate,"delayed":delayed,"immediate_prediction":pi,"delayed_prediction":pd,
                "global_regret":accept_loss-oracle,"immediate_regret":(accept_loss if pi>=0 else hold_loss)-oracle,
                "delayed_regret":(accept_loss if pd>=0 else hold_loss)-oracle})
    return events


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    commit = subprocess.check_output(["git","-C",str(ROOT/"external"/"matchmaker"),"rev-parse","HEAD"],text=True).strip()
    if commit != EXPECTED_COMMIT or digest(MODEL_PATH) != EXPECTED_MODEL:
        raise RuntimeError("frozen dependency drift")
    manifest = json.loads((AUDIO / "manifest.json").read_text(encoding="utf-8"))
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    files = [Path(__file__), PROTOCOL, MODEL_PATH, AUDIO/"manifest.json"]
    for spec in CASES:
        files += [AUDIO/f"{spec['stem']}.wav", DATA/"match"/f"{spec['stem']}.match", DATA/"musicxml"/spec["score"]]
    freeze = {str(p.relative_to(ROOT)).replace("\\","/"):digest(p) for p in files}
    OUT.mkdir(parents=True)
    write_json(OUT/"freeze.json",freeze); write_json(OUT/"state.json",{"status":"RUNNING"})
    processed=[]; implementation=True
    with threadpool_limits(limits=2):
        for number,spec in enumerate(CASES,1):
            print(f"[{number}/8] {spec['case']}",flush=True)
            score_path, audio_path = DATA/"musicxml"/spec["score"], AUDIO/f"{spec['stem']}.wav"
            score = merged_score(score_path)
            trackers={}; tempo=None
            prefix_path=OUT/f"{spec['case']}_prefix.wav"
            if spec["prefix_audit"]:
                samples,sr=sf.read(audio_path,always_2d=True)
                sf.write(prefix_path,samples[:int(round(20*sr))],sr,subtype=sf.info(audio_path).subtype)
            for method in METHODS:
                raw,runtime,current_tempo=run_tracker(score_path,audio_path,method)
                tempo = current_tempo if tempo is None else tempo
                if abs(current_tempo-tempo)>1e-9: raise RuntimeError("tracker tempo mismatch")
                beat_min,beat_max,score_duration,notes=score_geometry(score,tempo)
                valid=path_valid(raw,beat_min,beat_max); causal={"pass":True,"not_run":True}
                if spec["prefix_audit"]:
                    prefix,pruntime,ptempo=run_tracker(score_path,prefix_path,method)
                    causal=prefix_audit(raw,prefix); causal["runtime_sec"]=pruntime
                    if abs(ptempo-tempo)>1e-9: raise RuntimeError("prefix tempo mismatch")
                implementation=implementation and valid and bool(causal["pass"])
                raw_origin=float(score.inv_beat_map(beat_min)/score.quarter_duration_map(score.inv_beat_map(beat_min))*(60/tempo))
                converted=raw.copy(); converted[1]=score_seconds(score,raw[1],tempo,raw_origin)
                trackers[method]={"path_seconds":converted.tolist(),"path_points":int(raw.shape[1]),"runtime_sec":runtime,
                    "tempo":tempo,"path_valid":valid,"prefix_causality":causal}
                print(f"  {method}: points={raw.shape[1]} valid={valid} causal={causal['pass']}",flush=True)
            beat_min,beat_max,score_duration,notes=score_geometry(score,float(tempo))
            raw_origin=float(score.inv_beat_map(beat_min)/score.quarter_duration_map(score.inv_beat_map(beat_min))*(60/float(tempo)))
            query,truth,gt_audit=ground_truth(DATA/"match"/f"{spec['stem']}.match",score,float(tempo),raw_origin)
            score_chroma=notes_to_chroma_stream(notes,score_duration,HOP)
            case={"case":spec["case"],"work":spec["work"],"performer":spec["performer"],"audio":str(audio_path),
                "query_time":query.tolist(),"truth":truth.tolist(),"score_duration":score_duration,"score_chroma":score_chroma.tolist(),
                "score_domain_beats":[beat_min,beat_max],"gt_audit":gt_audit,"trackers":trackers}
            case["events"]=make_events(case,model) if implementation else []
            write_json(OUT/f"{spec['case']}.json",case); processed.append(case)
            print(f"  anchors={len(query)} events={len(case['events'])}",flush=True)
    events=[e for c in processed for e in c["events"]]; write_json(OUT/"events.json",events)
    works=sorted({e["work"] for e in events}); performances=sorted({e["case"] for e in events}); methods=sorted({e["method"] for e in events})
    per_work=[]
    for work in works:
        subset=[e for e in events if e["work"]==work]
        per_work.append({"work":work,"events":len(subset),"performances":len({e['case'] for e in subset}),
            "global_regret":float(np.mean([e["global_regret"] for e in subset])),
            "immediate_regret":float(np.mean([e["immediate_regret"] for e in subset])),
            "delayed_regret":float(np.mean([e["delayed_regret"] for e in subset]))})
    diff=np.asarray([r["global_regret"]-r["delayed_regret"] for r in per_work])
    ci=bootstrap_ci(diff) if len(diff) else [None,None]
    holds=[e for e in events if e["delayed_prediction"]<0]
    precision=float(np.mean([e["oracle_action"]=="hold" for e in holds])) if holds else None
    gates={"implementation":bool(implementation),"support":len(events)>=20 and len(performances)>=6 and len(works)>=3 and methods==list(METHODS),
        "work_equal_improvement":bool(len(diff) and diff.mean()>0),"bootstrap_lower_positive":bool(len(diff) and ci[0]>0),
        "no_material_work_worsening":bool(len(diff) and np.all(-diff<=.05+1e-12)),
        "hold_precision":bool(len(holds)>=5 and precision is not None and precision>=.8)}
    gates["pass"]=all(gates.values())
    nonzero=diff[np.abs(diff)>1e-12]
    exact=None
    if len(nonzero):
        means=[np.mean(nonzero*np.asarray(signs)) for signs in itertools.product((-1,1),repeat=len(nonzero))]
        exact=float(np.mean(np.asarray(means)>=nonzero.mean()-1e-15))
    result={"status":"PASS" if gates["pass"] else "FAIL","scope":"frozen Vienna 4x22 audio shadow transfer",
        "events":len(events),"performances":len(performances),"works":len(works),"methods":methods,"holds":len(holds),
        "hold_precision":precision,"work_equal_improvement":float(diff.mean()) if len(diff) else None,"bootstrap95":ci,
        "exact_sign_flip_p_one_sided":exact,"per_work":per_work,"gates":gates}
    if any(digest(ROOT/rel)!=expected for rel,expected in freeze.items()): raise RuntimeError("input drift")
    write_json(OUT/"result.json",result); write_json(OUT/"state.json",{"status":"COMPLETE","pass":gates["pass"]})
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
