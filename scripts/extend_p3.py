"""Run every frozen P3 extension contrast, then evaluate on untouched compositions."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.ground_truth import piecewise_linear_gt
from score_following.prospective import ridge_predict, bootstrap_mean
from score_following.p3_extension import (cosine_cost, oltw_from_cost, bayes_from_cost,
    evidence_at, score_window_features, score_ambiguity_at, distractor_sign,
    distractor_costs, cue_costs, sign_flip_p, holm_adjust)

CONDITIONS = ["intact", "targeted", "scattered", "prevention", "prevention_sham", "recovery", "recovery_sham"]
TRACKERS = ["oltw", "bayes"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty output {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def diverse(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["composer"]].append(row)
    return [b[i] for i in range(max(map(len, buckets.values()))) for b in buckets.values() if i<len(b)]


def make_manifest(cfg):
    root = ROOT/cfg["asap_root"]
    meta = read_csv(root/"metadata.csv")
    by_perf = {r["midi_performance"]: r for r in meta}
    old = read_csv(ROOT/cfg["old_manifest"])
    annot = json.loads((root/"asap_annotations.json").read_text(encoding="utf-8"))
    cases, excluded = [], []
    seen_labels, seen_scores = set(), set()
    def eligibility(r):
        a = annot.get(r["midi_performance"], {})
        if not a.get("score_and_performance_aligned", False):
            return "not_aligned"
        beats = a.get("performance_beats", [])
        earliest_history = cfg["landmark_start_sec"]-5-cfg["hop_sec"]
        if not beats or min(beats)>earliest_history or max(beats)<cfg["duration_sec"]:
            return "annotation_coverage"
        if not (root/r["midi_performance"]).exists() or not (root/r["midi_score"]).exists():
            return "missing_midi"
        notes = load_notes(root/r["midi_performance"])
        if not notes or max(n[1] for n in notes)<cfg["duration_sec"]:
            return "short_performance"
        return None
    def add(r, cohort):
        label = r["composer"] + "/" + r["title"]
        cases.append(dict(case_id=f"c{len(cases):03}", cohort=cohort, composition=label,
                          performance=r["midi_performance"], score=r["midi_score"]))
        seen_labels.add(label)
        seen_scores.add(r["midi_score"])
    for item in old:
        r = by_perf[item["performance"]]
        reason = eligibility(r)
        if reason:
            raise ValueError(f"existing cohort unavailable: {r['midi_performance']} {reason}")
        add(r, "development")
    n_new = 0
    for r in diverse(meta):
        label = r["composer"] + "/" + r["title"]
        if label in seen_labels or r["midi_score"] in seen_scores:
            continue
        reason = eligibility(r)
        if reason:
            excluded.append(dict(performance=r["midi_performance"], composition=label, reason=reason))
            continue
        add(r, "holdout")
        n_new += 1
        if n_new == cfg["holdout_compositions"]:
            break
    if n_new != cfg["holdout_compositions"]:
        raise ValueError("not enough eligible holdout compositions")
    return cases, excluded, annot


def cached_original_errors(cfg):
    manifest = read_csv(ROOT/cfg["old_manifest"])
    source = ROOT/"results/p2_asap_maxsim_20260908_135300/frame_results.csv"
    frames = read_csv(source)
    out, cursor = {}, 0
    for m in manifest:
        n = int(m["cached_frames"])
        out[m["performance"]] = [(int(round(float(r["perf_time_sec"])/cfg["hop_sec"])),
                                    float(r["online_error_sec"])) for r in frames[cursor:cursor+n]
                                   if float(r["perf_time_sec"])<cfg["duration_sec"]-.2]
        cursor += n
    return out


def process_case(case, cfg, annot, old_errors, out):
    root, hop = ROOT/cfg["asap_root"], cfg["hop_sec"]
    pn, sn = load_notes(root/case["performance"]), load_notes(root/case["score"])
    pd, sd = max(n[1] for n in pn), max(n[1] for n in sn)
    hop_v = hop*sd/pd
    u = notes_to_chroma_stream(pn, cfg["duration_sec"], hop)
    v = notes_to_chroma_stream(sn, sd, hop_v)
    a = annot[case["performance"]]
    gt, gt_lo, gt_hi = piecewise_linear_gt(a["performance_beats"], a["midi_score_beats"])
    time = np.arange(len(u))*hop
    truth = gt(time)/hop_v
    base = cosine_cost(u, v)
    band = int(round(cfg["band_sec"]/hop))
    positions = {}
    def replay(name, cost):
        positions[(name, "oltw")] = oltw_from_cost(cost, band)
        b = cfg["bayes"]
        positions[(name, "bayes")] = bayes_from_cost(cost, b["transition"], b["emission_beta"], b["restart_probability"])
    replay("intact", base)
    errors = abs(positions[("intact", "oltw")]-truth)*hop_v
    audit_diffs = [abs(errors[i]-error) for i, error in old_errors.get(case["performance"], [])]
    if audit_diffs and max(audit_diffs)>1e-7:
        raise AssertionError(f"historical baseline mismatch: {case['performance']} {max(audit_diffs)}")
    ic = cfg["intervention"]
    onset, end = [int(round(ic[k]/hop)) for k in ("onset_sec", "end_sec")]
    guard = max(1, int(round(ic["protected_halfwidth_score_sec"]/hop_v)))
    radius = max(1, int(round(cfg["local_candidate_radius_sec"]/hop_v)))
    offset = distractor_sign(case["performance"])*int(round(ic["distractor_offset_score_sec"]/hop_v))
    seed = cfg["seed"]+int(case["case_id"][1:])
    target, scattered, diag = distractor_costs(base, truth, onset=onset, end=end,
        offset=offset, guard=guard, radius=radius, seed=seed)
    exclude = max(1, int(round(cfg["competitor_exclude_sec"]/hop_v)))
    changes = []
    for i in range(onset, end):
        j = int(round(truth[i]))
        candidates = np.arange(max(0, j-radius), min(len(v), j+radius+1))
        wrong = candidates[abs(candidates-j)>exclude]
        # Correct cost is unchanged; positive value = wrong-match margin weakened.
        changes.append(float(base[i, wrong].min()-target[i, wrong].min()))
    diag.update(manipulation_weakness_increase=float(np.mean(changes)),
                distractor_offset_frames=offset, audit_frames=len(audit_diffs),
                audit_max_difference_sec=max(audit_diffs, default=0.))
    replay("targeted", target)
    replay("scattered", scattered)
    del scattered
    for kind in ["prevention", "recovery"]:
        first, stop = [int(round(t/hop)) for t in ic[kind+"_interval_sec"]]
        for sham in [False, True]:
            cost = cue_costs(target, truth, first=first, end=stop, guard=guard,
                             strength=ic["cue_strength"], sham=sham)
            replay(kind+("_sham" if sham else ""), cost)
            del cost
    del target, base
    for (condition, tracker), estimated in positions.items():
        if not np.array_equal(estimated[:onset], positions[("intact", tracker)][:onset]):
            raise AssertionError("intervention changed the past")
    diag["pre_max_position_difference_frames"] = 0
    # Save compact complete state traces; all future endpoints can be audited.
    np.savez_compressed(out/"cases"/(case["case_id"]+".npz"),
        time_sec=time, truth_frames=truth, hop_score_sec=hop_v,
        **{c+"__"+t: positions[(c,t)] for c in CONDITIONS for t in TRACKERS})
    summaries = []
    c0, c1 = [int(round(t/hop)) for t in ic["cue_outcome_interval_sec"]]
    for c in CONDITIONS:
        for t in TRACKERS:
            err = abs(positions[(c,t)]-truth)*hop_v
            valid = (time>=gt_lo)&(time<=gt_hi)
            summaries.append(dict(**case, tracker=t, condition=c,
                during_error_sec=float(err[onset:end].mean()), after_error_sec=float(err[c0:c1].mean()),
                baseline_median_error_sec=float(np.median(err[valid]))))
    # Natural evidence is computed from unmodified inputs only.
    window_live = int(round(cfg["evidence_window_sec"]/hop))
    window_score = max(2, int(round(cfg["evidence_window_sec"]/hop_v)))
    score_features = score_window_features(v, window_score)
    evidence_cache = {}
    def measure(i):
        if i not in evidence_cache:
            ev = evidence_at(u, v, truth, i, window=window_live, exclude=exclude, radius=radius,
                             frontier=positions[("intact", "oltw")][i], band=band)
            ev["ambiguity"] = score_ambiguity_at(score_features, truth[i], window_score, exclude)
            evidence_cache[i] = ev
        return evidence_cache[i]
    landmarks = []
    for t in np.arange(cfg["landmark_start_sec"], cfg["landmark_stop_sec"]+.001, cfg["landmark_stride_sec"]):
        i = int(round(t/hop))-1  # completed live frame only
        p2, p5 = i-int(round(2/hop)), i-int(round(5/hop))
        ev, prev = measure(i), measure(p2)
        for tracker in TRACKERS:
            err = abs(positions[("intact", tracker)]-truth)*hop_v
            row = dict(**case, tracker=tracker, time_sec=float(t),
                log_error=float(np.log1p(err[i])), log_error_past2=float(np.log1p(err[p2])),
                log_error_past5=float(np.log1p(err[p5])), elapsed_min=float(t/60),
                past_tempo=float((truth[i]-truth[p2])*hop_v/2), **ev,
                ambiguity_change2=ev["ambiguity"]-prev["ambiguity"],
                weakness_change2=ev["weakness_local"]-prev["weakness_local"])
            for h in cfg["horizons_sec"]:
                row[f"target_{h:g}s"] = float(np.log1p(err[i+int(round(h/hop))])-np.log1p(err[i]))
            if not all(np.isfinite(row[k]) for k in ["ambiguity", "weakness_local", "weakness_change2"]):
                raise AssertionError("nonfinite primary evidence; do not silently omit")
            landmarks.append(row)
    return summaries, landmarks, dict(**case, **diag)


def independent_means(rows, value):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["composition"]].append(value(r))
    names = sorted(grouped)
    return names, np.array([np.mean(grouped[name]) for name in names])


def summarize_effect(values, cfg):
    return dict(mean=float(np.mean(values)), ci95=bootstrap_mean(values, cfg["seed"], cfg["n_bootstrap"]),
                positive=int(np.sum(np.asarray(values)>0)), compositions=len(values),
                p=sign_flip_p(values, cfg["seed"], cfg["n_sign_flips"]))


def analyze(cfg, out, cases, summaries, landmarks, diagnostics):
    metrics = {"natural": {}, "intervention": {}, "primary_tests": [], "baseline": {},
               "scope": "Frozen within-ASAP held-out compositions; oracle cost interventions, two causal trackers."}
    primary = metrics["primary_tests"]
    effects_rows = []
    # Relationship 1: no duplicated tracker rows; fixed landmarks per performance.
    correlations = []
    for case in cases:
        rs = [r for r in landmarks if r["case_id"]==case["case_id"] and r["tracker"]=="oltw"]
        aa, weak = np.array([r["ambiguity"] for r in rs]), np.array([r["weakness_local"] for r in rs])
        informative = np.std(aa)>0 and np.std(weak)>0
        rho = float(spearmanr(aa, weak).statistic) if informative else 0.
        correlations.append(dict(**case, rho=rho, informative=informative))
    for cohort in ["development", "holdout"]:
        names, values = independent_means([r for r in correlations if r["cohort"]==cohort], lambda r:r["rho"])
        result = summarize_effect(values, cfg)
        metrics["natural"][cohort+"_ambiguity_to_weakness"] = result
        if cohort=="holdout":
            primary.append(dict(name="A_to_weakness", **result))
        for name, value in zip(names, values):
            effects_rows.append(dict(cohort=cohort, comparison="A_to_weakness", tracker="shared", composition=name, effect=value))
    write_csv(out/"natural_correlations.csv", correlations)
    # Predictive coefficients and scalers learn only from the old cohort.
    base_keys = ["log_error", "log_error_past2", "log_error_past5", "elapsed_min", "past_tempo"]
    models = {"baseline":base_keys,
              "ambiguity":base_keys+["ambiguity", "ambiguity_change2"],
              "evidence":base_keys+["weakness_local", "weakness_change2"],
              "joint":base_keys+["ambiguity", "ambiguity_change2", "weakness_local", "weakness_change2"]}
    prediction_rows = []
    for tracker in TRACKERS:
        train = [r for r in landmarks if r["cohort"]=="development" and r["tracker"]==tracker]
        test = [r for r in landmarks if r["cohort"]=="holdout" and r["tracker"]==tracker]
        group_counts = {g:sum(r["composition"]==g for r in train) for g in {r["composition"] for r in train}}
        weights = np.array([1/group_counts[r["composition"]] for r in train])
        metrics["natural"][tracker] = {}
        for h in cfg["horizons_sec"]:
            y = np.array([r[f"target_{h:g}s"] for r in test])
            train_y = np.array([r[f"target_{h:g}s"] for r in train])
            predictions = {}
            for model, keys in models.items():
                x = np.array([[r[k] for k in keys] for r in train])
                xx = np.array([[r[k] for k in keys] for r in test])
                predictions[model] = ridge_predict(x, train_y, xx, weights, cfg["ridge_alpha"])
            base_loss = abs(y-predictions["baseline"])
            results = {}
            for model in models:
                loss = abs(y-predictions[model])
                group_names = sorted({r["composition"] for r in test})
                effects = np.array([np.mean((base_loss-loss)[[r["composition"]==g for r in test]]) for g in group_names])
                result = summarize_effect(effects, cfg)
                result.update(mae=float(loss.mean()), baseline_mae=float(base_loss.mean()),
                              relative_improvement_percent=float(100*effects.mean()/base_loss.mean()))
                results[model] = result
                if h==cfg["primary_horizon_sec"] and model=="evidence":
                    primary.append(dict(name=tracker+"_evidence_forecast", **result))
                for g, value in zip(group_names, effects):
                    effects_rows.append(dict(cohort="holdout", comparison=f"forecast_{h:g}s_{model}", tracker=tracker, composition=g, effect=value))
            for i, row in enumerate(test):
                prediction_rows.append(dict(case_id=row["case_id"], composition=row["composition"], tracker=tracker,
                    time_sec=row["time_sec"], horizon_sec=h, target=float(y[i]),
                    **{model:float(pred[i]) for model, pred in predictions.items()}))
            metrics["natural"][tracker][f"{h:g}s"] = results
    write_csv(out/"holdout_predictions.csv", prediction_rows)
    comparisons = {
        "target_vs_intact": ("targeted", "intact", "during_error_sec"),
        "target_vs_scattered": ("targeted", "scattered", "during_error_sec"),
        "prevention_vs_target": ("targeted", "prevention", "after_error_sec"),
        "prevention_vs_sham": ("prevention_sham", "prevention", "after_error_sec"),
        "recovery_vs_target": ("targeted", "recovery", "after_error_sec"),
        "recovery_vs_sham": ("recovery_sham", "recovery", "after_error_sec"),
    }
    for cohort in ["development", "holdout"]:
        metrics["intervention"][cohort] = {}
        metrics["baseline"][cohort] = {}
        for tracker in TRACKERS:
            rr = [r for r in summaries if r["cohort"]==cohort and r["tracker"]==tracker]
            lookup = {(r["case_id"], r["condition"]):r for r in rr}
            case_items = [r for r in cases if r["cohort"]==cohort]
            metrics["intervention"][cohort][tracker] = {}
            _, baseline = independent_means([r for r in rr if r["condition"]=="intact"], lambda r:r["baseline_median_error_sec"])
            metrics["baseline"][cohort][tracker] = dict(mean_piece_median_sec=float(baseline.mean()),
                                                       median_piece_median_sec=float(np.median(baseline)))
            for name, (left, right, value_key) in comparisons.items():
                perfs = [dict(**r, difference=lookup[(r["case_id"], left)][value_key]-lookup[(r["case_id"], right)][value_key]) for r in case_items]
                names, effects = independent_means(perfs, lambda r:r["difference"])
                result = summarize_effect(effects, cfg)
                metrics["intervention"][cohort][tracker][name] = result
                if cohort=="holdout":
                    primary.append(dict(name=tracker+"_"+name, **result))
                for g, value in zip(names, effects):
                    effects_rows.append(dict(cohort=cohort, comparison=name, tracker=tracker, composition=g, effect=value))
    if len(primary)!=15:
        raise AssertionError("primary multiplicity family changed")
    adjusted = holm_adjust([r["p"] for r in primary])
    for row, adj in zip(primary, adjusted):
        row.update(holm_p=float(adj), passed=bool(row["mean"]>0 and row["ci95"][0]>0 and adj<=.05))
    metrics["audits"] = dict(
        historical_frames=sum(r["audit_frames"] for r in diagnostics),
        max_historical_error_difference_sec=max(r["audit_max_difference_sec"] for r in diagnostics),
        pre_max_difference_frames=max(r["pre_max_position_difference_frames"] for r in diagnostics),
        mean_weakness_manipulation=float(np.mean([r["manipulation_weakness_increase"] for r in diagnostics])),
        max_matched_budget_difference=max(r["max_row_sum_mismatch"] for r in diagnostics))
    write_csv(out/"composition_effects.csv", effects_rows)
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/p3_extension.yaml")
    args = p.parse_args()
    cfg_path = ROOT/args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = ROOT/"results"/f"{cfg['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out/"cases").mkdir()
    (out/"source_snapshot").mkdir()
    log_file = open(out/"run.log", "w", encoding="utf-8")
    def log(s):
        print(s, flush=True)
        log_file.write(s+"\n")
        log_file.flush()
    log(f"output: {out}")
    shutil.copy2(cfg_path, out/"config.yaml")
    cases, excluded, annot = make_manifest(cfg)
    write_csv(out/"manifest.csv", cases)
    if excluded:
        write_csv(out/"eligibility_exclusions.csv", excluded)
    sources = [cfg_path, Path(__file__), ROOT/"P3_EXTENSION_PROTOCOL.md",
        ROOT/"src/score_following/p3_extension.py", ROOT/"src/score_following/prospective.py",
        ROOT/"src/score_following/features.py", ROOT/"src/score_following/ground_truth.py"]
    files = sources + [ROOT/cfg["asap_root"]/r[k] for r in cases for k in ["performance", "score"]]
    for src in sources:
        shutil.copy2(src, out/"source_snapshot"/src.name)
    hashes = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/"provenance.json").write_text(json.dumps(dict(python=sys.version, numpy=np.__version__, sha256=hashes), indent=2), encoding="utf-8")
    log(f"frozen manifest: {len(cases)} performances; {sum(c['cohort']=='holdout' for c in cases)} new compositions")
    old_errors = cached_original_errors(cfg)
    summaries, landmarks, diagnostics = [], [], []
    with threadpool_limits(limits=1):
        for i, case in enumerate(cases):
            ss, ll, dd = process_case(case, cfg, annot, old_errors, out)
            summaries.extend(ss)
            landmarks.extend(ll)
            diagnostics.append(dd)
            write_csv(out/"performance_results.csv", summaries)
            write_csv(out/"landmarks.csv", landmarks)
            write_csv(out/"diagnostics.csv", diagnostics)
            log(f"{i+1}/{len(cases)} {case['cohort']} {case['performance']}: complete")
        metrics = analyze(cfg, out, cases, summaries, landmarks, diagnostics)
    for test in metrics["primary_tests"]:
        log(f"{test['name']}: mean={test['mean']:+.6f} CI={test['ci95']} Holm={test['holm_p']:.5f} pass={test['passed']}")
    log(json.dumps(metrics["audits"]))
    log_file.close()


if __name__ == "__main__":
    main()
