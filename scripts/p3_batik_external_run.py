"""Run the frozen Batik external temporal evaluation exactly once."""

import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import partitura as pt
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from score_following.temporal import healthy_before
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import predict_prior
from p3_short_signal import sustained_start

OUT = ROOT / "results/p3_batik_external"
CONFIG = ROOT / "configs/p3_temporal.yaml"
MATCHED = re.compile(r"^snote\(([^,]+),.*?,(-?\d+\.\d+),(-?\d+\.\d+),\[[^]]*\]\)-note\([^,]+,(\d+),(\d+),")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gt(path):
    text = path.read_text(encoding="utf-8")
    units = int(re.search(r"info\(midiClockUnits,(\d+)\)", text).group(1))
    rate = int(re.search(r"info\(midiClockRate,(\d+)\)", text).group(1))
    seconds_per_tick = rate / (units * 1_000_000)
    by_beat = defaultdict(list)
    for line in text.splitlines():
        match = MATCHED.match(line)
        if match:
            _, start, _, _, tick = match.groups()
            by_beat[float(start)].append(int(tick) * seconds_per_tick)
    beats = np.array(sorted(by_beat), dtype=float)
    perf_times = np.array([np.median(by_beat[b]) for b in beats])
    if len(beats) < 2 or np.any(np.diff(perf_times) <= 0):
        raise AssertionError(f"invalid GT clock: {path}")
    return beats, perf_times


def run_case(case, cfg):
    performance = ROOT / case["performance"]
    score_path = ROOT / case["score"]
    match_path = ROOT / case["match"]
    hop = float(cfg["hop_sec"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(score_path))
        score_notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_duration = float((score_notes["onset_beat"] + score_notes["duration_beat"]).max()) * .5
    performance_notes = load_notes(performance)
    performance_duration = max(n[1] for n in performance_notes)
    score_hop = hop * score_duration / performance_duration
    score_stream_notes = [(float(r["onset_beat"]) * .5,
                           float(r["onset_beat"] + r["duration_beat"]) * .5,
                           int(r["pitch"])) for r in score_notes]
    u = notes_to_chroma_stream(performance_notes, 90, hop)
    v = notes_to_chroma_stream(score_stream_notes, score_duration, score_hop)
    beats, perf_times = load_gt(match_path)
    time = np.arange(len(u)) * hop
    truth = np.interp(time, perf_times, beats * .5) / score_hop
    cost = cosine_cost(u, v)
    bayes = cfg["bayes"]
    path = bayes_from_cost(cost, bayes["transition"], bayes["emission_beta"], bayes["restart_probability"])
    error = np.abs(path - truth) * score_hop
    landmarks = [int(round(t / hop)) for t in np.arange(20, 75.001, 5)]
    landmarks = [i for i in landmarks if healthy_before(error, i - 1, 20, .5)]
    needed = {j for i in landmarks for j in range(i - 20, i)}
    posterior = np.zeros(len(v)); posterior[0] = 1
    state_index = np.arange(len(v))
    signals = {}
    identity = 0.
    for i, row in enumerate(cost):
        if i:
            posterior = predict_prior(posterior, bayes["transition"], bayes["restart_probability"])
        if i in needed:
            posterior, metrics = current_evidence(
                posterior, row, np.abs(state_index - truth[i]) * score_hop <= .5,
                bayes["emission_beta"])
            signals[i] = metrics["B"]
            identity = max(identity, abs(metrics["identity_residual"]))
        else:
            posterior *= np.exp(-bayes["emission_beta"] * (row - row.min()))
            posterior /= posterior.sum()
        if int(posterior.argmax()) != int(path[i]):
            raise AssertionError(f"path mismatch {case['sonata']}-{case['movement']} frame {i}")
    rows = []
    for i in landmarks:
        values = np.array([signals[j] for j in range(i - 20, i)])
        onset = sustained_start(error, i, i + 20)
        one = bool(np.any(values > 0))
        two = bool(np.any((values[:-1] > 0) & (values[1:] > 0)))
        rows.append({"sonata": case["sonata"], "movement": case["movement"],
                     "time_sec": i * hop, "sustained": onset is not None,
                     "onset_sec": onset * hop if onset is not None else None,
                     "confirmation_sec": (onset + 4) * hop if onset is not None else None,
                     "warning_available_sec": i * hop if two else None,
                     "any_positive": one, "two_consecutive": two,
                     "mean_B": float(values.mean()), "max_B": float(values.max())})
    return rows, identity


def confusion(rows, field):
    tp = sum(r[field] and r["sustained"] for r in rows)
    fp = sum(r[field] and not r["sustained"] for r in rows)
    fn = sum(not r[field] and r["sustained"] for r in rows)
    tn = sum(not r[field] and not r["sustained"] for r in rows)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "ppv": tp / (tp + fp) if tp + fp else None}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = ROOT / "P3_BATIK_EXTERNAL_PROTOCOL.md"
    if manifest["protocol_sha256"] != sha256(protocol):
        raise AssertionError("protocol changed after manifest freeze")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg["hop_sec"] != .1 or cfg["duration_sec"] != 90:
        raise AssertionError("frozen frame clock changed")
    rows, residuals = [], []
    eligible = [r for r in manifest["rows"] if r["eligible"]]
    with threadpool_limits(limits=1):
        for number, case in enumerate(eligible, 1):
            case_rows, residual = run_case(case, cfg)
            rows.extend(case_rows); residuals.append(residual)
            print(f"[{number}/{len(eligible)}] {case['sonata']}-{case['movement']}: {len(case_rows)} landmarks")
    event_onsets = {(r["sonata"], r["movement"], r["onset_sec"]) for r in rows if r["sustained"]}
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row["sonata"], row["movement"])].append(row)
    event_cases = {case for case, rr in by_case.items() if any(r["sustained"] for r in rr)}
    nonevent_cases = set(by_case) - event_cases
    warned_event_cases = {case for case in event_cases if any(r["two_consecutive"] and r["sustained"] for r in by_case[case])}
    warned_nonevent_cases = {case for case in nonevent_cases if any(r["two_consecutive"] for r in by_case[case])}
    event_count = len(event_onsets)
    event_capture = len(warned_event_cases) / len(event_cases) if event_cases else None
    nonevent_fpr = len(warned_nonevent_cases) / len(nonevent_cases) if nonevent_cases else None
    all_preceded = all(r["warning_available_sec"] is not None and r["warning_available_sec"] < r["onset_sec"]
                       for r in rows if r["sustained"])
    if event_count < 5:
        decision = "insufficient_information"
    elif all_preceded and event_capture >= .8 and nonevent_fpr <= .2:
        decision = "temporal_support_signal"
    else:
        decision = "negative"
    summary = {
        "scope": "frozen external temporal evaluation; one composer and performer",
        "eligible_movements": len(eligible), "eligible_sonatas": len({r["sonata"] for r in eligible}),
        "eligible_landmarks": len(rows), "unique_event_onsets": event_count,
        "event_movements": len(event_cases), "nonevent_movements": len(nonevent_cases),
        "primary_two_consecutive": confusion(rows, "two_consecutive"),
        "secondary_any_positive": confusion(rows, "any_positive"),
        "event_movement_capture": event_capture,
        "nonevent_movement_false_alarm": nonevent_fpr,
        "all_event_landmark_warnings_precede_onset": all_preceded,
        "decision": decision,
        "max_bayes_identity_residual": max(residuals, default=0.),
        "provenance": {"manifest_sha256": sha256(manifest_path), "protocol_sha256": sha256(protocol),
                       "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__))},
        "rows": rows,
        "caveat": "This single-composer/single-performer cohort cannot establish broad natural P3 generality."
    }
    (OUT / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
