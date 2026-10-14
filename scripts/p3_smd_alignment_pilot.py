"""Development-only SMD alignment feasibility; never use its output as GT."""

import hashlib
import json
from pathlib import Path

import numpy as np
import partitura as pt

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "asap_dataset/Bach/Fugue/bwv_875/xml_score.musicxml"
PERF = ROOT / "data/smd_direct_midi/Bach_BWV875-02_002_20090916-SMD.mid"
OUT = ROOT / "results/p3_external_metadata/smd_bwv875_alignment_pilot.json"


def events(notes, time_field, threshold=0.0):
    """Group simultaneous score notes or performance notes within a short onset window."""
    ordered = sorted(notes, key=lambda x: float(x[time_field]))
    groups = []
    for note in ordered:
        time = float(note[time_field])
        if not groups or time - groups[-1][0] > threshold:
            groups.append((time, set()))
        groups[-1][1].add(int(note["pitch"]))
    return groups


def align(score_events, perf_events, chroma=False):
    n, m = len(score_events), len(perf_events)
    cost = np.full((n + 1, m + 1), np.inf)
    step = np.zeros((n + 1, m + 1), dtype=np.uint8)
    cost[0, 0] = 0.0
    score_sets = [{p % 12 for p in pitches} if chroma else pitches
                  for _, pitches in score_events]
    perf_sets = [{p % 12 for p in pitches} if chroma else pitches
                 for _, pitches in perf_events]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = score_sets[i - 1], perf_sets[j - 1]
            distance = 1.0 - len(a & b) / len(a | b)
            candidates = (cost[i - 1, j - 1], cost[i - 1, j] + .2,
                          cost[i, j - 1] + .2)
            direction = int(np.argmin(candidates))
            cost[i, j] = distance + candidates[direction]
            step[i, j] = direction
    pairs = []
    i, j = n, m
    while i and j:
        pairs.append((i - 1, j - 1))
        direction = step[i, j]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    by_perf = {}
    for si, pj in pairs:
        by_perf.setdefault(pj, []).append(si)
    mapped = {pj: int(np.median(indices)) for pj, indices in by_perf.items()}
    return cost[n, m] / len(pairs), mapped, len(pairs)


def main():
    score = pt.load_musicxml(SCORE).note_array()
    perf = pt.load_performance_midi(PERF).note_array()
    se = events(score, "onset_quarter")
    pe = events(perf, "onset_sec", threshold=.045)
    exact_cost, exact, exact_steps = align(se, pe)
    chroma_cost, chroma, chroma_steps = align(se, pe, chroma=True)
    common = sorted(exact.keys() & chroma.keys())
    disagreements = [abs(se[exact[j]][0] - se[chroma[j]][0]) for j in common]
    pitch_jaccard = [len(se[exact[j]][1] & pe[j][1]) / len(se[exact[j]][1] | pe[j][1])
                     for j in sorted(exact)]
    low_agreement = [{"performance_sec": pe[j][0], "score_quarter": se[exact[j]][0],
                      "performance_pitches": sorted(pe[j][1]),
                      "score_pitches": sorted(se[exact[j]][1]),
                      "pitch_jaccard": pitch_jaccard[j]}
                     for j in sorted(exact) if pitch_jaccard[j] < .5]
    anchors = [{"performance_sec": pe[j][0], "score_quarter": se[exact[j]][0],
                "pitch_jaccard": pitch_jaccard[j]} for j in range(0, len(pe), max(1, len(pe) // 10))]
    result = {
        "score": str(SCORE.relative_to(ROOT)),
        "performance": str(PERF.relative_to(ROOT)),
        "score_sha256": hashlib.sha256(SCORE.read_bytes()).hexdigest(),
        "performance_sha256": hashlib.sha256(PERF.read_bytes()).hexdigest(),
        "score_notes": len(score), "performance_notes": len(perf),
        "score_onset_events": len(se), "performance_onset_events": len(pe),
        "methods": {"exact_pitch": {"mean_path_cost": exact_cost, "steps": exact_steps},
                    "chroma": {"mean_path_cost": chroma_cost, "steps": chroma_steps}},
        "method_agreement": {"mapped_performance_events": len(common),
                             "median_score_quarter_difference": float(np.median(disagreements)),
                             "p95_score_quarter_difference": float(np.percentile(disagreements, 95)),
                             "events_disagree_gt_1_quarter": sum(d > 1 for d in disagreements)},
        "exact_pitch_diagnostics": {
            "score_events_covered": len(set(exact.values())),
            "median_pitch_jaccard": float(np.median(pitch_jaccard)),
            "p05_pitch_jaccard": float(np.percentile(pitch_jaccard, 5)),
            "events_pitch_jaccard_below_half": sum(v < .5 for v in pitch_jaccard),
            "first_last_score_quarter": [se[exact[min(exact)]][0], se[exact[max(exact)]][0]],
            "first_last_performance_sec": [pe[min(exact)][0], pe[max(exact)][0]],
            "anchors": anchors},
        "low_pitch_agreement_events": low_agreement,
        "caveat": "Both alignments are automatic and related DTW methods. Agreement is a feasibility diagnostic, not independent ground truth or P3 outcome.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
