#!/usr/bin/env python3
"""Synthetic and open-file calibration of two outcome-blind event aligners."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

try:
    from scripts import p3_natural_boundary_review_packet as packet
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_boundary_review_packet as packet
    import p3_natural_event_restart_candidates as event

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_natural_alignment_calibration.json"
GAP_COST = 0.35
SYNTHETIC_SEEDS = 20


@njit(cache=False)
def popcount(value):
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=False)
def jaccard_distance(a_lo, a_hi, b_lo, b_hi):
    intersection = popcount(a_lo & b_lo) + popcount(a_hi & b_hi)
    union = popcount(a_lo | b_lo) + popcount(a_hi | b_hi)
    return 1.0 - intersection / union if union else 0.0


@njit(cache=False)
def align_masks(score_lo, score_hi, perf_lo, perf_hi, gap_cost=GAP_COST):
    n, m = len(score_lo), len(perf_lo)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    previous = np.arange(m + 1, dtype=np.float64) * gap_cost
    trace[0, 1:] = 2
    for i in range(1, n + 1):
        current = np.empty(m + 1, dtype=np.float64); current[0] = i * gap_cost; trace[i, 0] = 1
        for j in range(1, m + 1):
            diagonal = previous[j-1] + jaccard_distance(score_lo[i-1], score_hi[i-1], perf_lo[j-1], perf_hi[j-1])
            up = previous[j] + gap_cost; left = current[j-1] + gap_cost
            if diagonal <= up and diagonal <= left:
                current[j] = diagonal; trace[i, j] = 0
            elif up <= left:
                current[j] = up; trace[i, j] = 1
            else:
                current[j] = left; trace[i, j] = 2
        previous = current
    mapping = np.full(m, -1, dtype=np.int64); i, j = n, m
    while i or j:
        direction = trace[i, j]
        if i and j and direction == 0:
            mapping[j-1] = i-1; i -= 1; j -= 1
        elif i and (not j or direction == 1):
            i -= 1
        else:
            j -= 1
    return mapping


def masks(groups, chroma=False):
    lo, hi = [], []
    for group in groups:
        pitches = group["pitches"]
        if chroma: pitches = {pitch % 12 for pitch in pitches}
        low = high = 0
        for pitch in set(pitches):
            if pitch < 64: low |= 1 << pitch
            else: high |= 1 << (pitch - 64)
        lo.append(low); hi.append(high)
    return np.asarray(lo, dtype=np.uint64), np.asarray(hi, dtype=np.uint64)


def align(score_events, performance_events, chroma=False):
    score_lo, score_hi = masks(score_events, chroma); perf_lo, perf_hi = masks(performance_events, chroma)
    return align_masks(score_lo, score_hi, perf_lo, perf_hi)


def synthetic_performance(score_events, seed):
    rng = np.random.default_rng(seed); output = []; truth = []
    for index, group in enumerate(score_events):
        if rng.random() < .05: continue
        pitches = list(group["pitches"])
        if rng.random() < .05 and pitches:
            selected = int(rng.integers(len(pitches))); pitches[selected] = int(np.clip(pitches[selected] + rng.choice((-1, 1)), 0, 127))
        output.append({"pitches": pitches, "time_sec": float(len(output)), "latest_end_sec": float(len(output))+.1}); truth.append(index)
        if rng.random() < .05:
            inserted = list(pitches); inserted[0] = int(np.clip(inserted[0] + rng.choice((-2, 2)), 0, 127))
            output.append({"pitches": inserted, "time_sec": float(len(output)), "latest_end_sec": float(len(output))+.1}); truth.append(-1)
    return output, np.asarray(truth)


def percentile(values, q): return float(np.percentile(values, q)) if len(values) else float("inf")


def max_unmapped_seconds(mapping, events):
    maximum = 0.0; start = None
    for index, value in enumerate(mapping):
        if value < 0 and start is None: start = events[index]["time_sec"]
        if value >= 0 and start is not None:
            maximum = max(maximum, events[index]["time_sec"] - start); start = None
    if start is not None: maximum = max(maximum, events[-1]["latest_end_sec"] - start)
    return float(maximum)


def synthetic_runs():
    acquisition = json.loads((ROOT/"results/p3_pianovam_acquisition.json").read_text(encoding="utf8"))
    unique = {}
    for row in acquisition["rows"]: unique.setdefault(str(row["rule_index"]), row)
    selected = [unique[key] for key in sorted(unique, key=lambda value: int(value))[:8]]
    rows = []
    for work_number, record in enumerate(selected):
        score_notes = event.load_notes(ROOT/record["score_midi_path"])
        score_events = event.onset_events(score_notes, event.SCORE_ONSET_TOLERANCE)
        for seed in range(SYNTHETIC_SEEDS):
            performance, truth = synthetic_performance(score_events, 1000*work_number+seed)
            exact = align(score_events, performance, False); chroma = align(score_events, performance, True)
            known = truth >= 0; exact_error = np.abs(exact[known]-truth[known]); both = (exact >= 0) & (chroma >= 0)
            disagreement = np.abs(exact[both]-chroma[both])
            rows.append({"work_id":record["rule_index"],"seed":seed,
                "exact_median_error":percentile(exact_error,50),"exact_p95_error":percentile(exact_error,95),
                "method_p95_disagreement":percentile(disagreement,95)})
    return rows


def pilot_records():
    records={row["record_time"]:row for row in packet.all_records()}
    return [records[key] for key in ("SMD_Brahms_Op010No1_003_20090916","MAESTRO_Liszt_Gnomenreigen_2011","MAESTRO_Scriabin_Sonata5_2008")]


def real_pilots():
    rows=[]
    for record in pilot_records():
        score=event.onset_events(event.load_notes(ROOT/record["score_midi_path"]),event.SCORE_ONSET_TOLERANCE)
        performance=event.onset_events(event.load_notes(ROOT/record["performance_path"]),event.PERFORMANCE_ONSET_TOLERANCE)
        exact=align(score,performance,False);chroma=align(score,performance,True);both=(exact>=0)&(chroma>=0)
        disagreement=np.abs(exact[both]-chroma[both]);mapped_exact=float(np.mean(exact>=0));mapped_chroma=float(np.mean(chroma>=0))
        metrics={"record_time":record["record_time"],"score_events":len(score),"performance_events":len(performance),
            "exact_mapped_share":mapped_exact,"chroma_mapped_share":mapped_chroma,
            "median_method_disagreement":percentile(disagreement,50),"p95_method_disagreement":percentile(disagreement,95),
            "exact_max_unmapped_sec":max_unmapped_seconds(exact,performance),"chroma_max_unmapped_sec":max_unmapped_seconds(chroma,performance)}
        gates={"exact_mapped_ge_0_80":mapped_exact>=.80,"chroma_mapped_ge_0_80":mapped_chroma>=.80,
            "median_disagreement_le_2":metrics["median_method_disagreement"]<=2,
            "p95_disagreement_le_20":metrics["p95_method_disagreement"]<=20,
            "exact_unmapped_le_5sec":metrics["exact_max_unmapped_sec"]<=5,
            "chroma_unmapped_le_5sec":metrics["chroma_max_unmapped_sec"]<=5}
        rows.append({**metrics,"gates":gates,"pass":all(gates.values())})
    return rows


def main():
    synthetic=synthetic_runs();n=len(synthetic)
    synthetic_gates={"median_error_pass_share_ge_0_95":sum(r["exact_median_error"]<=1 for r in synthetic)/n>=.95,
        "p95_error_pass_share_ge_0_95":sum(r["exact_p95_error"]<=5 for r in synthetic)/n>=.95,
        "method_agreement_pass_share_ge_0_95":sum(r["method_p95_disagreement"]<=10 for r in synthetic)/n>=.95}
    pilots=real_pilots();pilot_gate=sum(r["pass"] for r in pilots)>=2
    result={"scope":"Outcome-blind alignment calibration; no tracker/P3 outcomes",
        "parameters":{"gap_cost":GAP_COST,"synthetic_seeds_per_work":SYNTHETIC_SEEDS,"works":8,
          "deletion_rate":.05,"insertion_rate":.05,"pitch_alteration_rate":.05},
        "synthetic_summary":{"runs":n,"median_error_pass_share":sum(r["exact_median_error"]<=1 for r in synthetic)/n,
          "p95_error_pass_share":sum(r["exact_p95_error"]<=5 for r in synthetic)/n,
          "method_agreement_pass_share":sum(r["method_p95_disagreement"]<=10 for r in synthetic)/n,
          "gates":synthetic_gates},"real_pilots":pilots,"real_pilot_gate_two_of_three":pilot_gate,
        "continue_after_certification":all(synthetic_gates.values()) and pilot_gate,"alignment_is_ground_truth":False,
        "validation_eligible":False,"synthetic_runs":synthetic}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({key:value for key,value in result.items() if key!="synthetic_runs"},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
