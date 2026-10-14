"""One-shot Batik development evaluation of trajectory-coherent evidence."""

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
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior

OUT = ROOT / "results/p3_coherent_trajectory"
SOURCE = ROOT / "results/p3_batik_external"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_COHERENT_TRAJECTORY_PROTOCOL.md"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shift_weights(weights, steps):
    result = np.zeros_like(weights)
    if steps == 0:
        return weights.copy()
    if steps >= len(weights):
        result[-1] = weights.sum()
    else:
        result[steps:] = weights[:-steps]
        result[-1] += weights[-steps:].sum()
    return result


def choose_regions(prior, radius, exclude):
    n = len(prior); index = np.arange(n)
    current_center = int(np.argmax(prior))
    current = np.abs(index - current_center) <= radius
    prefix = np.concatenate(([0.], np.cumsum(prior)))
    lo = np.maximum(0, index - radius); hi = np.minimum(n, index + radius + 1)
    masses = prefix[hi] - prefix[lo]
    eligible = np.abs(index - current_center) > exclude + radius
    if not eligible.any():
        raise ValueError("no disjoint competitor")
    competitor_center = int(np.flatnonzero(eligible)[np.argmax(masses[eligible])])
    competitor = np.abs(index - competitor_center) <= radius
    wc = prior * current; ww = prior * competitor
    wc /= wc.sum(); ww /= ww.sum()
    return wc, ww, current_center, competitor_center


def cumulative_evidence(prior, cost_window, radius, exclude, beta, expected_step):
    current, competitor, cc, cw = choose_regions(prior, radius, exclude)
    total = 0.; contributions = []
    for offset, cost in enumerate(cost_window):
        likelihood = np.exp(-beta * (cost - cost.min()))
        if np.all(likelihood == likelihood[0]):
            value = 0.
        else:
            step = int(round(offset * expected_step))
            lc = float(shift_weights(current, step) @ likelihood)
            lw = float(shift_weights(competitor, step) @ likelihood)
            value = float(np.log(lw) - np.log(lc))
        contributions.append(value); total += value
    return total, contributions, cc, cw


def load_case(case, cfg):
    performance = ROOT / case["performance"]
    score_path = ROOT / case["score"]
    hop = float(cfg["hop_sec"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(score_path))
        notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_duration = float((notes["onset_beat"] + notes["duration_beat"]).max()) * .5
    pnotes = load_notes(performance)
    perf_duration = max(n[1] for n in pnotes)
    score_hop = hop * score_duration / perf_duration
    snotes = [(float(r["onset_beat"]) * .5,
               float(r["onset_beat"] + r["duration_beat"]) * .5,
               int(r["pitch"])) for r in notes]
    return (notes_to_chroma_stream(pnotes, 90, hop),
            notes_to_chroma_stream(snotes, score_duration, score_hop), score_hop)


def confusion(rows):
    tp = sum(r["warning"] and r["sustained"] for r in rows)
    fp = sum(r["warning"] and not r["sustained"] for r in rows)
    fn = sum(not r["warning"] and r["sustained"] for r in rows)
    tn = sum(not r["warning"] and not r["sustained"] for r in rows)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / (tp + fn), "false_alarm_rate": fp / (fp + tn),
            "ppv": tp / (tp + fp) if tp + fp else None}


def rank_auc(positive, negative):
    comparisons = [1. if p > n else .5 if p == n else 0. for p in positive for n in negative]
    return float(np.mean(comparisons))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((SOURCE / "result.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    labels = {(r["sonata"], r["movement"], r["time_sec"]): r for r in old["rows"]}
    cases = [r for r in manifest["rows"] if r["eligible"]]
    bayes = cfg["bayes"]; transition = np.asarray(bayes["transition"], dtype=float)
    expected_step = float(np.arange(len(transition)) @ transition)
    rows = []
    with threadpool_limits(limits=1):
        for number, case in enumerate(cases, 1):
            relevant = [r for k, r in labels.items() if k[:2] == (case["sonata"], case["movement"])]
            if not relevant:
                continue
            u, v, score_hop = load_case(case, cfg)
            cost = cosine_cost(u, v)
            query_frames = {int(round(r["time_sec"] / cfg["hop_sec"])) for r in relevant}
            starts = {i - 20 for i in query_frames}
            priors = {}
            posterior = np.zeros(len(v)); posterior[0] = 1
            for i, cost_row in enumerate(cost):
                if i:
                    posterior = predict_prior(posterior, transition, bayes["restart_probability"])
                if i in starts:
                    priors[i] = posterior.copy()
                likelihood = np.exp(-bayes["emission_beta"] * (cost_row - cost_row.min()))
                posterior *= likelihood; posterior /= posterior.sum()
            radius = max(1, int(round(.5 / score_hop)))
            exclude = max(1, int(round(2. / score_hop)))
            for label in relevant:
                end = int(round(label["time_sec"] / cfg["hop_sec"])); start = end - 20
                value, parts, current, competitor = cumulative_evidence(
                    priors[start], cost[start:end], radius, exclude,
                    bayes["emission_beta"], expected_step)
                rows.append({"sonata": case["sonata"], "movement": case["movement"],
                             "time_sec": label["time_sec"], "sustained": label["sustained"],
                             "onset_sec": label["onset_sec"], "C": value, "warning": value > 0,
                             "current_center_start": current, "competitor_center_start": competitor,
                             "zero_contribution_frames": sum(x == 0 for x in parts)})
            print(f"[{number}/{len(cases)}] {case['sonata']}-{case['movement']}")
    table = confusion(rows)
    by_case = defaultdict(list)
    for row in rows: by_case[(row["sonata"], row["movement"])].append(row)
    event_cases = {k for k, rr in by_case.items() if any(r["sustained"] for r in rr)}
    nonevent_cases = set(by_case) - event_cases
    warned_event = {k for k in event_cases if any(r["warning"] and r["sustained"] for r in by_case[k])}
    warned_nonevent = {k for k in nonevent_cases if any(r["warning"] for r in by_case[k])}
    unique_events = len({(r["sonata"], r["movement"], r["onset_sec"]) for r in rows if r["sustained"]})
    prevalence = sum(r["sustained"] for r in rows) / len(rows)
    case_fpr = len(warned_nonevent) / len(nonevent_cases)
    lead_ok = all(r["time_sec"] < r["onset_sec"] for r in rows if r["warning"] and r["sustained"])
    gate = (unique_events >= 5 and table["sensitivity"] >= .8 and table["false_alarm_rate"] <= .2
            and table["ppv"] is not None and table["ppv"] >= 2 * prevalence
            and case_fpr <= .3 and lead_ok)
    event_values = [r["C"] for r in rows if r["sustained"]]
    nonevent_values = [r["C"] for r in rows if not r["sustained"]]
    result = {"scope": "one-shot Batik development gate; not independent confirmation",
              "signal": "trajectory-coherent cumulative observation log likelihood ratio C>0",
              "landmarks": len(rows), "unique_event_onsets": unique_events,
              "confusion": table, "event_prevalence": prevalence,
              "event_movement_capture": len(warned_event) / len(event_cases),
              "nonevent_movement_false_alarm": case_fpr,
              "captured_warnings_precede_onset": lead_ok, "gate_pass": gate,
              "decision": "advance_to_independent_gt_build" if gate else "stop_signal_family",
              "C_median_event": float(np.median(event_values)),
              "C_median_nonevent": float(np.median(nonevent_values)),
              "C_rank_auc_diagnostic": rank_auc(event_values, nonevent_values),
              "rows": rows,
              "provenance": {"protocol_sha256": sha256(PROTOCOL), "config_sha256": sha256(CONFIG),
                             "manifest_sha256": sha256(SOURCE / "manifest.json"),
                             "labels_sha256": sha256(SOURCE / "result.json"),
                             "script_sha256": sha256(Path(__file__))}}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
