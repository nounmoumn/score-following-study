"""Frozen phase-3 contrast: filtering vs fixed-lag smoothing on ONE model.

Follows CONFIDENCE_SMOOTHING_V1_PROTOCOL.md. Only the amount of future observation
changes. Smoothing reads future performance frames, so no number here is a live
result, a latency claim, or a deployable method.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.fixed_lag_smoothing import backward_adjoint, smoothed_trace
from score_following.p3_extension import bayes_from_cost, cosine_cost

CONFIG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/confidence_smoothing_v1.json")
PROTOCOL = ROOT / "CONFIDENCE_SMOOTHING_V1_PROTOCOL.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def weighted_auc(score, y, w):
    """P(score of a correct row > score of a wrong row), ties counted at one half."""
    order = np.argsort(score, kind="stable")
    s, y, w = score[order], y[order], w[order]
    pos, neg = w*y, w*(1-y)
    total, seen = 0., 0.
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        tie_pos, tie_neg = pos[i:j].sum(), neg[i:j].sum()
        total += tie_pos*(seen + tie_neg/2)
        seen += tie_neg
        i = j
    denominator = pos.sum()*neg.sum()
    return float(total/denominator) if denominator else None


def common_anchors(perf_beats, score_beats, frames, hop, max_lag):
    """Beats every lag can answer, so all conditions are scored on the SAME beats.

    A larger lag has no estimate for the last max_lag frames, and at emission time it
    cannot answer the first max_lag beats either. Comparing lags on whatever each can
    reach would confound the contrast with a different evaluation set, so the window
    is fixed once, from the largest lag, and shared.
    """
    perf_beats, score_beats = np.asarray(perf_beats, float), np.asarray(score_beats, float)
    first, last = (max_lag+1)*hop, (frames-max_lag)*hop
    mask = (perf_beats >= first) & (perf_beats <= last)
    return perf_beats[mask], score_beats[mask]


def anchor(trace, perf_beats, score_beats, hop, tolerance, lag, *, charge_delay):
    """Pick one estimate per shared beat.

    charge_delay=True selects by emission time (estimated frame + lag), which is what
    an operator would actually see, so a larger lag pays its wait. charge_delay=False
    selects the estimate FOR the frame containing the beat regardless of when it was
    emitted; that isolates the value of future observations from the cost of waiting
    and is never an online result.
    """
    latest = np.floor(perf_beats/hop).astype(int)-1        # last frame ending at or before
    idx = latest - (lag if charge_delay else 0)
    if np.any(idx < 0) or np.any(idx >= len(trace["position_score_sec"])):
        raise ValueError(f"shared anchor outside estimate range for lag {lag}")
    emitted_at = (idx + (lag if charge_delay else 0) + 1)*hop
    error = np.abs(trace["position_score_sec"][idx]-score_beats)
    return {"indices": idx, "times": perf_beats, "truth": score_beats,
            "ages": perf_beats-emitted_at, "error": error,
            "correct": (error <= tolerance+1e-12).astype(int)}


def work_weights(rows):
    counts = Counter((r["work_family"], r["case_id"]) for r in rows)
    families = defaultdict(set)
    for family, case in counts:
        families[family].add(case)
    w = np.array([1./len(families)/len(families[r["work_family"]])/
                  counts[(r["work_family"], r["case_id"])] for r in rows])
    return w/w.sum()


def summarize(rows, bins=10):
    if not rows:
        return {"observations": 0}
    w = work_weights(rows)
    micro = np.full(len(rows), 1./len(rows))
    c = np.array([r["confidence"] for r in rows], float)
    y = np.array([r["correct"] for r in rows], float)
    out = {"observations": len(rows), "cases": len({r["case_id"] for r in rows}),
           "works": len({r["work_family"] for r in rows})}
    for label, weight in (("work_equal", w), ("micro", micro)):
        clipped = np.clip(c, 1e-12, 1-1e-12)
        ids = np.minimum((c*bins).astype(int), bins-1)
        ece = 0.
        for i in range(bins):
            mask = ids == i
            if not mask.any():
                continue
            mass = float(weight[mask].sum())
            ece += mass*abs(float(weight[mask] @ c[mask]/mass) - float(weight[mask] @ y[mask]/mass))
        out[label] = {"accuracy": float(weight @ y), "mean_confidence": float(weight @ c),
                      "brier": float(weight @ ((c-y)**2)),
                      "logloss": float(-weight @ (y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
                      "ece": float(ece), "auc": weighted_auc(c, y, weight)}
    out["mean_output_age_sec"] = float(np.mean([r["output_age_sec"] for r in rows]))
    out["max_output_age_sec"] = float(np.max([r["output_age_sec"] for r in rows]))
    return out


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["out"]
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    started = time.perf_counter()
    data = ROOT / cfg["asap_root"]
    with (ROOT / cfg["source_manifest"]).open(encoding="utf-8-sig", newline="") as f:
        cases = [r for r in csv.DictReader(f) if r["role"] == cfg["role"]]
    if len(cases) != cfg["expected_performances"]:
        raise AssertionError("unexpected cohort count")
    if cfg.get("limit_cases"):     # smoke runs only; recorded in the freeze
        cases = cases[:cfg["limit_cases"]]
    annotations = json.loads((data / "asap_annotations.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / cfg["audit_run"] / "result.json").read_text(encoding="utf-8"))
    audit_accuracy = audit["primary"]["work_equal"]["accuracy"]

    # The backward operator must preserve the all-ones vector, or the kernel is wrong.
    ones_residual = max(float(np.abs(backward_adjoint(np.ones(n), cfg["transition"],
                                                      cfg["restart"]) - 1).max())
                        for n in (16, 256, 4096))
    if ones_residual > 1e-12:
        raise AssertionError(f"backward operator does not preserve ones: {ones_residual}")

    out.mkdir(parents=True)
    frozen = [CONFIG, PROTOCOL, Path(__file__),
              ROOT / "src/score_following/fixed_lag_smoothing.py",
              ROOT / "src/score_following/observation_evidence.py",
              ROOT / "src/score_following/p3_extension.py",
              ROOT / "tests/test_fixed_lag_smoothing.py",
              ROOT / cfg["audit_run"] / "result.json"]
    freeze = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "previously exposed development; no independent validation",
              "not_a_live_result": "smoothing reads future performance frames",
              "python": platform.python_version(), "numpy": np.__version__, "config": cfg,
              "backward_ones_residual": ones_residual,
              "audit_reference_accuracy": audit_accuracy,
              "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in frozen}}
    write_json(out / "freeze.json", freeze)
    write_json(out / "state.json", {"status": "RUNNING", "freeze_sha256": digest(out/"freeze.json")})

    hp, hs = cfg["performance_hop_sec"], cfg["score_hop_sec"]
    tolerance = cfg["position_tolerance_score_sec"]
    options = dict(score_hop=hs, tolerance=tolerance, transition=cfg["transition"],
                   beta=cfg["beta"], restart=cfg["restart"])
    primary = {str(lag): [] for lag in cfg["lags"]}
    delay_free = {str(lag): [] for lag in cfg["lags"]}
    summaries, failures, score_cache = [], [], {}
    with threadpool_limits(limits=cfg["blas_threads"]):
        for number, case in enumerate(cases, 1):
            case_start = time.perf_counter()
            try:
                ann = annotations[case["performance"]]
                if not ann.get("score_and_performance_aligned", False):
                    raise ValueError("annotation not aligned")
                pnotes = load_notes(data / case["performance"])
                key = case["score_sha256"]
                if key not in score_cache:
                    snotes = load_notes(data / case["score"])
                    score_cache[key] = notes_to_chroma_stream(snotes, max(n[1] for n in snotes), hs)
                v = score_cache[key]
                frames = int(np.floor(min(cfg["replay_prefix_sec"],
                                          max(n[1] for n in pnotes))/hp))
                if frames <= max(cfg["lags"]):
                    raise ValueError("prefix shorter than the largest lag")
                u = notes_to_chroma_stream(pnotes, frames*hp, hp)[:frames]
                cost = cosine_cost(u, v)
                reference = bayes_from_cost(cost, cfg["transition"], cfg["beta"], cfg["restart"])
                shared_beats, shared_truth = common_anchors(
                    ann["performance_beats"], ann["midi_score_beats"], frames, hp,
                    max(cfg["lags"]))
                if len(shared_beats) < cfg["min_shared_anchors"]:
                    raise ValueError(f"only {len(shared_beats)} shared anchors")
                entry = {"case_id": case["case_id"], "work_family": case["work_family"],
                         "frames": frames, "score_states": len(v),
                         "shared_anchors": int(len(shared_beats)), "conditions": {}}
                for lag in cfg["lags"]:
                    trace = smoothed_trace(cost, lag, **options)
                    path_equal = bool(np.array_equal(trace["map_index"], reference[:len(trace["map_index"])]))
                    if lag == 0 and not path_equal:
                        raise AssertionError("lag 0 does not reproduce the audit filter")
                    for label, store, charge in (("primary", primary, True),
                                                 ("delay_free", delay_free, False)):
                        labels = anchor(trace, shared_beats, shared_truth, hp, tolerance,
                                        lag, charge_delay=charge)
                        for k, i in enumerate(labels["indices"]):
                            store[str(lag)].append({
                                "case_id": case["case_id"], "work_family": case["work_family"],
                                "lag": lag, "performance_time_sec": float(labels["times"][k]),
                                "output_age_sec": float(labels["ages"][k]),
                                "position_score_sec": float(trace["position_score_sec"][i]),
                                "truth_score_sec": float(labels["truth"][k]),
                                "confidence": float(trace["confidence"][i]),
                                "pmax": float(trace["pmax"][i]),
                                "error_score_sec": float(labels["error"][k]),
                                "correct": int(labels["correct"][k])})
                    entry["conditions"][str(lag)] = {
                        "estimates": int(len(trace["map_index"])),
                        "emission_delay_sec": lag*hp,
                        "matches_filter_path_on_shared_prefix": path_equal}
                entry["runtime_sec"] = time.perf_counter()-case_start
                summaries.append(entry)
                print(f"[{number}/{len(cases)}] {case['case_id']} frames={frames} "
                      f"states={len(v)} {entry['runtime_sec']:.1f}s", flush=True)
            except Exception as error:
                failure = {"case_id": case["case_id"], "error": str(error),
                           "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(out / (case["case_id"]+"_failure.json"), failure)
                print(f"[{number}/{len(cases)}] {case['case_id']} FAILED: {error}", flush=True)

    metrics = {"primary_emission_time": {k: summarize(v, cfg["reliability_bins"])
                                         for k, v in primary.items()},
               "delay_free_diagnostic": {k: summarize(v, cfg["reliability_bins"])
                                         for k, v in delay_free.items()}}
    base = str(cfg["lags"][0])
    top = str(cfg["lags"][-1])
    gates = {}
    filter_accuracy = metrics["primary_emission_time"][base]["work_equal"]["accuracy"]
    gates["1_equivalence"] = {
        "lag0_work_equal_accuracy": filter_accuracy,
        "audit_reference": audit_accuracy,
        "difference": filter_accuracy-audit_accuracy,
        "pass": bool(abs(filter_accuracy-audit_accuracy) < 1e-9)}
    free_base = metrics["delay_free_diagnostic"][base]["work_equal"]["accuracy"]
    free_top = metrics["delay_free_diagnostic"][top]["work_equal"]["accuracy"]
    gates["2_information_restriction"] = {
        "lag0_delay_free_accuracy": free_base, "lag_top_delay_free_accuracy": free_top,
        "absolute_gain": free_top-free_base, "threshold": cfg["gate_delay_free_accuracy"],
        "pass": bool(free_top-free_base >= cfg["gate_delay_free_accuracy"])}
    auc_base = metrics["delay_free_diagnostic"][base]["work_equal"]["auc"]
    auc_top = metrics["delay_free_diagnostic"][top]["work_equal"]["auc"]
    gates["3_ordering"] = {
        "lag0_auc": auc_base, "lag_top_auc": auc_top, "absolute_gain": auc_top-auc_base,
        "threshold": cfg["gate_auc_gain"],
        "pass": bool(auc_top is not None and auc_base is not None
                     and auc_top-auc_base >= cfg["gate_auc_gain"])}
    primary_top = metrics["primary_emission_time"][top]["work_equal"]["accuracy"]
    gates["4_delay_honesty"] = {
        "lag0_primary_accuracy": filter_accuracy, "lag_top_primary_accuracy": primary_top,
        "absolute_gain": primary_top-filter_accuracy,
        "waiting_costs_what_it_buys": bool(primary_top <= filter_accuracy),
        "note": "no latency benefit is claimed in any case"}

    result = {"scope": freeze["scope"], "not_a_live_result": freeze["not_a_live_result"],
              "fresh_holdout": False, "independent_confirmation": False,
              "expected_cases": len(cases), "completed_cases": len(summaries),
              "completed_work_families": len({s["work_family"] for s in summaries}),
              "failures": failures, "backward_ones_residual": ones_residual,
              "metrics": metrics, "gates": gates, "case_summaries": summaries,
              "wall_sec": time.perf_counter()-started,
              "freeze_sha256": digest(out/"freeze.json"),
              "decision": "smoothing_contrast_complete" if not failures
                          else "incomplete_inspect_failures"}
    write_json(out / "result.json", result)
    for label, store in (("primary", primary), ("delay_free", delay_free)):
        rows = [r for v in store.values() for r in v]
        if rows:
            with (out / f"{label}_beats.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    write_json(out / "state.json", {"status": "COMPLETE" if not failures else "INCOMPLETE",
                                    "result_sha256": digest(out/"result.json"),
                                    "decision": result["decision"]})
    table = {lag: {"primary_acc": metrics["primary_emission_time"][lag]["work_equal"]["accuracy"],
                   "delay_free_acc": metrics["delay_free_diagnostic"][lag]["work_equal"]["accuracy"],
                   "delay_free_auc": metrics["delay_free_diagnostic"][lag]["work_equal"]["auc"],
                   "delay_free_ece": metrics["delay_free_diagnostic"][lag]["work_equal"]["ece"]}
             for lag in metrics["primary_emission_time"]}
    print(json.dumps({"completed": len(summaries), "failures": len(failures),
                      "by_lag": table, "gates": {k: v.get("pass", v) for k, v in gates.items()}},
                     indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
