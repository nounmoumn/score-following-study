"""Run the frozen KRAISLER transfer-development gate for coherent evidence."""

import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import partitura as pt
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from score_following.temporal import healthy_before
from score_following.observation_evidence import predict_prior
from p3_short_signal import sustained_start
from p3_batik_external_run import load_gt
from p3_coherent_trajectory import cumulative_evidence, rank_auc

OUT = ROOT / "results/p3_kraisler_signal"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_KRAISLER_SIGNAL_PROTOCOL.md"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(case, hop):
    performance = ROOT / case["performance"]; score_path = ROOT / case["score"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(score_path))
        notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_duration = float((notes["onset_beat"] + notes["duration_beat"]).max()) * .5
    pnotes = load_notes(performance); duration = max(n[1] for n in pnotes)
    score_hop = hop * score_duration / duration
    snotes = [(float(r["onset_beat"]) * .5,
               float(r["onset_beat"] + r["duration_beat"]) * .5,
               int(r["pitch"])) for r in notes]
    return (notes_to_chroma_stream(pnotes, duration, hop),
            notes_to_chroma_stream(snotes, score_duration, score_hop),
            score_hop, duration)


def confusion(rows):
    values = {}
    for warning, event, name in ((True, True, "tp"), (True, False, "fp"),
                                 (False, True, "fn"), (False, False, "tn")):
        values[name] = sum(r["warning"] == warning and r["sustained"] == event for r in rows)
    values.update(sensitivity=values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else None,
                  false_alarm_rate=values["fp"] / (values["fp"] + values["tn"]) if values["fp"] + values["tn"] else None,
                  ppv=values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else None)
    return values


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL): raise AssertionError("protocol changed after freeze")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); hop = float(cfg["hop_sec"])
    bayes = cfg["bayes"]; transition = np.asarray(bayes["transition"], dtype=float)
    expected_step = float(np.arange(len(transition)) @ transition)
    rows = []; case_summaries = []
    eligible = [r for r in manifest["rows"] if r["eligible"]]
    with threadpool_limits(limits=1):
        for number, case in enumerate(eligible, 1):
            u, v, score_hop, duration = load_case(case, hop)
            beats, perf_times = load_gt(ROOT / case["match"])
            time = np.arange(len(u)) * hop
            truth = np.interp(time, perf_times, beats * .5) / score_hop
            cost = cosine_cost(u, v)
            path = bayes_from_cost(cost, transition, bayes["emission_beta"], bayes["restart_probability"])
            error = np.abs(path - truth) * score_hop
            candidate_times = np.arange(15., duration - 5. + 1e-9, 5.)
            landmarks = [int(round(t / hop)) for t in candidate_times]
            landmarks = [i for i in landmarks if i + 20 < len(error) and healthy_before(error, i - 1, 20, .5)]
            starts = {i - 20 for i in landmarks}; priors = {}
            posterior = np.zeros(len(v)); posterior[0] = 1
            for i, cost_row in enumerate(cost):
                if i: posterior = predict_prior(posterior, transition, bayes["restart_probability"])
                if i in starts: priors[i] = posterior.copy()
                likelihood = np.exp(-bayes["emission_beta"] * (cost_row - cost_row.min()))
                posterior *= likelihood; posterior /= posterior.sum()
                if int(np.argmax(posterior)) != int(path[i]): raise AssertionError(f"path mismatch {case['track']} {i}")
            radius = max(1, int(round(.5 / score_hop))); exclude = max(1, int(round(2. / score_hop)))
            case_rows = []
            for end in landmarks:
                start = end - 20
                value, parts, current, competitor = cumulative_evidence(
                    priors[start], cost[start:end], radius, exclude,
                    bayes["emission_beta"], expected_step)
                onset = sustained_start(error, end, end + 20)
                row = {"track": case["track"], "composer": case["composer"], "title": case["title"],
                       "time_sec": end * hop, "C": value, "warning": value > 0,
                       "sustained": onset is not None, "onset_sec": onset * hop if onset is not None else None,
                       "current_center_start": current, "competitor_center_start": competitor,
                       "zero_contribution_frames": sum(x == 0 for x in parts)}
                rows.append(row); case_rows.append(row)
            case_summaries.append({"track": case["track"], "landmarks": len(case_rows),
                                   "events": len({r["onset_sec"] for r in case_rows if r["sustained"]}),
                                   "mean_C": float(np.mean([r["C"] for r in case_rows])) if case_rows else None,
                                   "event_case": any(r["sustained"] for r in case_rows)})
            print(f"[{number}/{len(eligible)}] {case['track']}: {len(case_rows)} landmarks")
    event = [r["C"] for r in rows if r["sustained"]]; nonevent = [r["C"] for r in rows if not r["sustained"]]
    usable_cases = [r for r in case_summaries if r["landmarks"]]
    event_case_values = [r["mean_C"] for r in usable_cases if r["event_case"]]
    nonevent_case_values = [r["mean_C"] for r in usable_cases if not r["event_case"]]
    unique_events = len({(r["track"], r["onset_sec"]) for r in rows if r["sustained"]})
    landmark_auc = rank_auc(event, nonevent) if event and nonevent else None
    case_auc = rank_auc(event_case_values, nonevent_case_values) if event_case_values and nonevent_case_values else None
    gate = (unique_events >= 5 and len(event_case_values) >= 5 and np.median(event) > np.median(nonevent)
            and landmark_auc >= .7 and case_auc >= .7)
    result = {"scope": "KRAISLER transfer development, not P3 confirmation",
              "eligible_tracks": len(eligible), "usable_tracks": len(usable_cases), "landmarks": len(rows),
              "unique_event_onsets": unique_events, "event_tracks": len(event_case_values),
              "confusion_C_gt_0_secondary": confusion(rows),
              "C_median_event": float(np.median(event)) if event else None,
              "C_median_nonevent": float(np.median(nonevent)) if nonevent else None,
              "landmark_rank_auc": landmark_auc, "track_mean_rank_auc": case_auc,
              "gate_pass": bool(gate), "decision": "advance_to_smd_gt_build" if gate else "stop_signal_family",
              "case_summaries": case_summaries, "rows": rows,
              "provenance": {"protocol_sha256": sha256(PROTOCOL), "manifest_sha256": sha256(manifest_path),
                             "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__))}}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "case_summaries", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
