#!/usr/bin/env python3
"""P3-Rg: repeat-pass constrained recovery at outcome-blind structural boundaries."""

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from p3_r_candidate_audit import posterior_until
from p3_r_basin_recovery import burden, continue_path
from p3_r_trigger_audit import centers_and_masses, compare_features
from p3_r_repeat_mapping_audit import build_mapping

STRUCTURE = ROOT / "results/p3_r_repeat_structure_audit.json"
MAPPING = ROOT / "results/p3_r_repeat_mapping_audit.json"
ACQUISITION = ROOT / "results/p3_pianocore_p3r_acquisition.json"
CONFIG = ROOT / "configs/p3_temporal.yaml"
OUT = ROOT / "results/p3_r_repeat_graph"
HOP = .1; BETA = 8.; RESTART = 1e-6; RESTORE_AFTER_ENTRY = (2., 5.); SEED = 20260920


def label_at(times, occurrences, score_seconds):
    index = int(np.searchsorted(times, score_seconds, side="right") - 1)
    return int(occurrences[max(0, min(index, len(occurrences)-1))])


def bootstrap(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["composer"], []).append(row[key])
    values = np.asarray([np.mean(group) for group in grouped.values()])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(value) for value in np.quantile(draws, [.025, .975])]}


def main():
    structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    good = {row["id"] for row in mapping["rows"] if row["mapping_eligible"]}
    srows = {row["id"]: row for row in structure["rows"]}
    cases = {row["id"]: row for row in acquisition["rows"]}
    transition = np.asarray(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["bayes"]["transition"])
    rows = []
    with threadpool_limits(limits=1):
        for number, identifier in enumerate(sorted(good), 1):
            case, srow = cases[identifier], srows[identifier]
            map_metrics, times, occurrences = build_mapping(srow)
            entry_score_sec = map_metrics["pass2_entries_score_sec"][0]
            pnotes, snotes = load_notes(ROOT/case["performance_path"]), load_notes(ROOT/case["score_path"])
            pd, sd = max(note[1] for note in pnotes), max(note[1] for note in snotes)
            score_hop = HOP * sd / pd
            base = cosine_cost(notes_to_chroma_stream(pnotes, pd, HOP), notes_to_chroma_stream(snotes, sd, score_hop))
            intact = bayes_from_cost(base, transition, BETA, RESTART); end = len(base)
            entry_perf_sec = entry_score_sec * pd / sd
            for after in RESTORE_AFTER_ENTRY:
                restore = int(round((entry_perf_sec + after) / HOP)); onset = restore - int(round(10/HOP))
                decision, eval_end = restore + int(round(2/HOP)), restore + int(round(10/HOP))
                if onset < int(round(5/HOP)) or eval_end > end:
                    continue
                erased = base.copy(); erased[onset:restore] = erased[onset:restore].mean(axis=1, keepdims=True)
                outage = bool(np.all(np.std(erased[onset:restore], axis=1) <= 1e-12))
                intact_false = bool(np.all(np.std(base[onset:restore], axis=1) <= 1e-12))
                natural = bayes_from_cost(erased, transition, BETA, RESTART)
                nb = burden(natural, intact, score_hop, decision, eval_end)
                posterior, prefix = posterior_until(erased, restore, transition, BETA, RESTART)
                centers, _ = centers_and_masses(posterior, score_hop)
                states, _, scores, _, _, _ = compare_features(erased, posterior, centers, transition, score_hop, restore, decision)
                history = prefix[onset-int(round(5/HOP)):onset].astype(float)
                slope = float(np.clip(np.polyfit(np.arange(len(history)), history, 1)[0], 0, len(transition)-1))
                predicted = float(history[-1] + slope*(restore-onset))
                predicted_pass = label_at(times, occurrences, predicted*score_hop)
                anchor = int(history[-1])
                legal = [i for i, center in enumerate(centers)
                         if center >= anchor and label_at(times, occurrences, center*score_hop) == predicted_pass]
                selected = max(legal, key=lambda index: scores[index]) if legal else None
                burdens = []
                for state in states:
                    path = natural.copy(); path[decision:] = continue_path(erased, state.copy(), transition, decision, end)
                    burdens.append(burden(path, intact, score_hop, decision, eval_end))
                sb = burdens[selected] if selected is not None else nb
                ob = min([nb] + burdens)
                correct_pass = label_at(times, occurrences, int(intact[restore-1])*score_hop)
                rows.append({"id": identifier, "composer": case["composer"], "restore_after_entry_sec": after,
                             "entry_score_sec": entry_score_sec, "window_perf_sec": [onset*HOP, restore*HOP],
                             "outage_detected": outage, "intact_false_outage": intact_false,
                             "predicted_pass": predicted_pass, "correct_pass": correct_pass,
                             "pass_correct": predicted_pass == correct_pass, "legal_candidates": len(legal),
                             "acted": selected is not None, "persistent_event": nb > .5,
                             "natural_burden": nb, "selector_burden": sb, "oracle_burden": ob,
                             "selector_improvement": nb-sb, "oracle_improvement": nb-ob})
            print(f"[{number}/{len(good)}] {identifier}")
    metrics = {key: bootstrap(rows, key) for key in ("natural_burden", "selector_burden", "oracle_burden",
                                                      "selector_improvement", "oracle_improvement")}
    events = sum(row["persistent_event"] for row in rows)
    affected = len({row["composer"] for row in rows if row["persistent_event"]})
    recall = float(np.mean([row["pass_correct"] for row in rows]))
    safety = float(np.mean([row["intact_false_outage"] for row in rows]))
    improvement = metrics["selector_improvement"]
    gates = {"incidence": events >= 10 and affected >= 5, "anchor_recall": recall >= .80,
             "intact_safety": safety <= .05,
             "recovery": improvement["mean"] > 0 and improvement["ci95"][0] > 0}
    gates["all"] = all(gates.values())
    result = {"scope": "P3-Rg repeat-pass constrained known-outage development; not natural P3",
              "coverage": {"files": len(good), "interventions": len(rows)},
              "incidence": {"events": events, "affected_composers": affected},
              "pass_anchor_recall": recall, "intact_false_outage_rate": safety,
              "actions": sum(row["acted"] for row in rows),
              "helpful": sum(row["selector_improvement"] > 0 for row in rows),
              "harmful": sum(row["selector_improvement"] < 0 for row in rows),
              "metrics": metrics, "gates": gates,
              "decision": "repeat_graph_development_pass" if gates["all"] else "repeat_graph_not_supported",
              "preserved_validation_requested": 0, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
