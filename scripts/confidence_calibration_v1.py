"""Frozen phase-2 calibration: do A and e add over ordinary confidence signals?

Follows CONFIDENCE_CALIBRATION_V1_PROTOCOL.md. Reads only stored audit features, so
the tracking path is identical in every variant. Fits nothing on held-out labels,
selects thresholds from predictions alone, and refuses to overwrite a finished run.
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
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/confidence_calibration_v1.json")
PROTOCOL = ROOT / "CONFIDENCE_CALIBRATION_V1_PROTOCOL.md"
AUDIT_PROTOCOL = ROOT / "CONFIDENCE_AUDIT_V1_PROTOCOL.md"

# Columns that encode the answer. None may ever reach a feature matrix.
FORBIDDEN = {"correct", "error_score_sec", "truth_score_sec", "performance_time_sec",
             "ambiguity_group", "output_age_sec", "available_at_sec", "case_id",
             "work_family", "composer"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def load_rows(cfg):
    """Audit rows plus the two derived ambiguity terms. No label is touched here."""
    audit = ROOT / cfg["audit_run"] / "primary_beats.csv"
    rows = list(csv.DictReader(audit.open(encoding="utf-8")))
    profile = json.loads((ROOT / cfg["score_profile"]).read_text(encoding="utf-8"))
    with (ROOT / cfg["source_manifest"]).open(encoding="utf-8-sig", newline="") as f:
        score_of = {r["case_id"]: r["score_sha256"] for r in csv.DictReader(f)
                    if r["role"] == cfg["role"]}
    dropped = 0
    kept = []
    for row in rows:
        if float(row["evidence_saturated"]) == 1:
            dropped += 1
            continue
        # A_work comes from the score's own maxsim profile, never from evaluated rows:
        # which frames are evaluated is set by annotation times.
        work = profile[score_of[row["case_id"]]]["mean"]
        row["A_work"] = work
        row["A_frame"] = float(row["ambiguity"]) - work
        kept.append(row)
    return kept, dropped, audit


def work_weight_factors(rows):
    """Equal works, equal performances within work, equal observations within one."""
    counts = Counter((r["work_family"], r["case_id"]) for r in rows)
    families = defaultdict(set)
    for family, case in counts:
        families[family].add(case)
    factor = np.array([1./len(families[r["work_family"]])/counts[(r["work_family"], r["case_id"])]
                       for r in rows])
    order = sorted(families)
    index = np.array([order.index(r["work_family"]) for r in rows])
    return factor, index, order


def matched_threshold(p, order, weight, target):
    """Smallest score admitting at least `target` of the weight. Uses no labels."""
    cumulative = np.cumsum(weight[order])/weight.sum()
    k = int(np.searchsorted(cumulative, target - 1e-12))
    k = min(k, len(order)-1)
    threshold = p[order][k]
    selected = p >= threshold
    return threshold, selected


def selected_error(p, order, weight, y, target):
    _, selected = matched_threshold(p, order, weight, target)
    w = weight[selected]
    if not w.sum():
        return None, 0.
    return float(w @ (1-y[selected])/w.sum()), float(w.sum()/weight.sum())


def score_model(p, y, weight, cfg, order):
    w = weight/weight.sum()
    clipped = np.clip(p, 1e-12, 1-1e-12)
    bins = cfg["reliability_bins"]
    ids = np.minimum((p*bins).astype(int), bins-1)
    ece, reliability = 0., []
    for i in range(bins):
        mask = ids == i
        if not mask.any():
            continue
        mass = float(w[mask].sum())
        pred, acc = float(w[mask] @ p[mask]/mass), float(w[mask] @ y[mask]/mass)
        ece += mass*abs(pred-acc)
        reliability.append({"bin": i, "observations": int(mask.sum()), "weight": mass,
                            "confidence": pred, "accuracy": acc})
    out = {"brier": float(w @ ((p-y)**2)),
           "logloss": float(-w @ (y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
           "ece": float(ece), "accuracy": float(w @ y), "mean_prediction": float(w @ p),
           "reliability": reliability}
    error, coverage = selected_error(p, order, weight, y, cfg["matched_coverage"])
    out["selected_error_at_matched_coverage"] = error
    out["achieved_matched_coverage"] = coverage
    out["secondary_coverage"] = {}
    for target in cfg["secondary_coverage"]:
        err, cov = selected_error(p, order, weight, y, target)
        out["secondary_coverage"][str(target)] = {"selected_error": err, "achieved_coverage": cov}
    return out


def fit_predict(name, columns, rows, y, family_index, cfg):
    """Leave-one-work-family-out. C chosen inside training folds only."""
    x = np.array([[float(r[c]) for c in columns] for r in rows], dtype=float)
    p = np.full(len(rows), np.nan)
    chosen, folds = [], []
    for held in np.unique(family_index):
        test = family_index == held
        train = ~test
        groups = family_index[train]
        best, best_score = cfg["regularization_grid"][0], -np.inf
        splits = min(cfg["inner_folds"], len(np.unique(groups)))
        for c_value in cfg["regularization_grid"]:
            total, count = 0., 0
            for inner_train, inner_test in GroupKFold(n_splits=splits).split(
                    x[train], y[train], groups):
                xi, yi = x[train][inner_train], y[train][inner_train]
                if len(np.unique(yi)) < 2:
                    continue
                mean, sd = xi.mean(0), xi.std(0)
                sd = np.where(sd > 0, sd, 1.)
                model = LogisticRegression(C=c_value, max_iter=cfg["max_iter"],
                                           random_state=cfg["random_state"])
                model.fit((xi-mean)/sd, yi)
                xo = (x[train][inner_test]-mean)/sd
                q = np.clip(model.predict_proba(xo)[:, 1], 1e-12, 1-1e-12)
                yo = y[train][inner_test]
                total += float(np.mean(yo*np.log(q)+(1-yo)*np.log1p(-q)))
                count += 1
            if count and total/count > best_score:
                best, best_score = c_value, total/count
        mean, sd = x[train].mean(0), x[train].std(0)
        sd = np.where(sd > 0, sd, 1.)
        model = LogisticRegression(C=best, max_iter=cfg["max_iter"],
                                   random_state=cfg["random_state"])
        model.fit((x[train]-mean)/sd, y[train])
        p[test] = model.predict_proba((x[test]-mean)/sd)[:, 1]
        chosen.append(best)
        folds.append({"family_index": int(held), "C": best,
                      "train_observations": int(train.sum()),
                      "test_observations": int(test.sum())})
    if not np.isfinite(p).all():
        raise AssertionError(f"{name}: unpredicted rows remain")
    return p, {"features": columns, "folds": folds,
               "selected_C_counts": {str(c): chosen.count(c) for c in sorted(set(chosen))}}


def bootstrap(models, y, factor, family_index, families, cfg, orders):
    """Paired percentile interval over work families. Threshold is re-derived each time."""
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(families)
    baseline = cfg["baseline_group"]
    draws = {name: [] for name in models}
    for _ in range(cfg["bootstrap_resamples"]):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        weight = counts[family_index]*factor
        if weight.sum() <= 0:
            continue
        values = {}
        for name, p in models.items():
            err, _ = selected_error(p, orders[name], weight, y, cfg["matched_coverage"])
            values[name] = err
        if values[baseline] is None:
            continue
        for name, value in values.items():
            draws[name].append(np.nan if value is None else value - values[baseline])
    low = (1-cfg["bootstrap_interval"])/2
    out = {}
    for name, series in draws.items():
        series = np.array([v for v in series if np.isfinite(v)])
        out[name] = {"resamples": int(len(series)),
                     "paired_difference_vs_baseline_median": float(np.median(series)),
                     "interval_low": float(np.quantile(series, low)),
                     "interval_high": float(np.quantile(series, 1-low)),
                     "excludes_zero": bool(np.quantile(series, low) > 0
                                           or np.quantile(series, 1-low) < 0)}
    return out


def gates(metrics, intervals, cfg):
    base = metrics[cfg["baseline_group"]]["selected_error_at_matched_coverage"]
    verdicts = {}
    best_ece = min(m["ece"] for m in metrics.values())
    verdicts["1_necessity"] = {
        "best_ece": best_ece, "threshold": cfg["gate_ece_max"],
        "pass": bool(best_ece <= cfg["gate_ece_max"]),
        "baseline_raw_confidence_ece": 0.172215}
    candidates = {}
    for name in cfg["method_claim_groups"]:
        error = metrics[name]["selected_error_at_matched_coverage"]
        candidates[name] = {"selected_error": error,
                            "relative_reduction_vs_B": (base-error)/base if base else None,
                            "interval_excludes_zero": intervals[name]["excludes_zero"]}
    best = max(candidates, key=lambda k: candidates[k]["relative_reduction_vs_B"])
    verdicts["2_method_claim"] = {
        "baseline_selected_error": base, "candidates": candidates, "best": best,
        "best_relative_reduction": candidates[best]["relative_reduction_vs_B"],
        "threshold": cfg["gate_method_relative"],
        "pass": bool(candidates[best]["relative_reduction_vs_B"] >= cfg["gate_method_relative"]
                     and candidates[best]["interval_excludes_zero"])}
    cheap = metrics["B+F"]["selected_error_at_matched_coverage"]
    best_gain = base - candidates[best]["selected_error"]
    cheap_gain = base - cheap
    share = cheap_gain/best_gain if best_gain > 0 else None
    beats = (cheap - candidates[best]["selected_error"])/cheap if cheap else None
    verdicts["3_cheap_competitor"] = {
        "flat_only_selected_error": cheap, "cheap_gain": cheap_gain, "best_gain": best_gain,
        "cheap_share_of_best_gain": share, "share_threshold": cfg["gate_cheap_share"],
        "best_relative_reduction_vs_cheap": beats,
        "threshold_vs_cheap": cfg["gate_beat_cheap_relative"],
        "cheap_explains_most_of_gain": bool(share is not None and share >= cfg["gate_cheap_share"]),
        "method_beats_cheap": bool(beats is not None and beats >= cfg["gate_beat_cheap_relative"])}
    within = metrics["B+Af"]["selected_error_at_matched_coverage"]
    verdicts["4_ambiguity_per_frame"] = {
        "B+Af_selected_error": within,
        "relative_reduction_vs_B": (base-within)/base if base else None,
        "threshold": cfg["gate_within_score_relative"],
        "pass": bool(base and (base-within)/base >= cfg["gate_within_score_relative"]
                     and intervals["B+Af"]["excludes_zero"])}
    verdicts["5_estimation_not_operation"] = {
        "note": "operator burden is never claimed; no human study exists",
        "estimation_improved": bool(best_ece < 0.172215),
        "operation_improved": verdicts["2_method_claim"]["pass"]}
    return verdicts


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["out"]
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {out}")
    started = time.perf_counter()
    rows, dropped, audit_csv = load_rows(cfg)
    y = np.array([float(r["correct"]) for r in rows])
    factor, family_index, families = work_weight_factors(rows)
    weight = factor/len(families)
    weight = weight/weight.sum()

    checks = {"saturated_rows_dropped": dropped, "observations": len(rows),
              "work_families": len(families),
              "performances": len({r["case_id"] for r in rows}),
              "label_column": "correct", "label_mean": float(y.mean())}
    used = sorted({c for cols in cfg["feature_groups"].values() for c in cols})
    leaked = sorted(set(used) & FORBIDDEN)
    checks["feature_columns"] = used
    checks["forbidden_columns_in_features"] = leaked
    if leaked:
        raise AssertionError(f"ground-truth columns in features: {leaked}")
    # A work family must never straddle a fold boundary.
    checks["families_span_one_fold_only"] = all(
        len({family_index[i] for i, r in enumerate(rows) if r["work_family"] == f}) == 1
        for f in families)
    if not checks["families_span_one_fold_only"]:
        raise AssertionError("work family split across folds")

    out.mkdir(parents=True)
    frozen = [CONFIG, PROTOCOL, AUDIT_PROTOCOL, Path(__file__), audit_csv,
              ROOT / cfg["score_profile"], ROOT / "scripts/score_ambiguity_profile.py",
              ROOT / "src/score_following/confidence_audit.py",
              ROOT / cfg["audit_run"] / "result.json", ROOT / cfg["audit_run"] / "freeze.json"]
    freeze = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "previously exposed development; no independent validation",
              "python": platform.python_version(), "numpy": np.__version__,
              "sklearn": sklearn.__version__,
              "config": cfg, "checks_before_outcomes": checks,
              "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in frozen}}
    write_json(out / "freeze.json", freeze)
    write_json(out / "state.json", {"status": "RUNNING", "freeze_sha256": digest(out/"freeze.json")})

    failures = []
    predictions, diagnostics = {}, {}
    with threadpool_limits(limits=cfg["blas_threads"]):
        for name, columns in cfg["feature_groups"].items():
            try:
                p, info = fit_predict(name, columns, rows, y, family_index, cfg)
                predictions[name], diagnostics[name] = p, info
                print(f"{name}: done, C counts {info['selected_C_counts']}", flush=True)
            except Exception as error:
                failures.append({"group": name, "error": str(error),
                                 "traceback": traceback.format_exc()})
                print(f"{name} FAILED: {error}", flush=True)
    if not predictions:
        raise SystemExit("no group produced predictions")

    orders = {name: np.argsort(-p, kind="stable") for name, p in predictions.items()}
    metrics = {name: score_model(p, y, weight, cfg, orders[name])
               for name, p in predictions.items()}
    intervals = bootstrap(predictions, y, factor, family_index, families, cfg, orders)
    verdicts = gates(metrics, intervals, cfg) if not failures else None

    raw = np.array([float(r["confidence"]) for r in rows])
    raw_order = np.argsort(-raw, kind="stable")
    raw_error, raw_coverage = selected_error(raw, raw_order, weight, y, cfg["matched_coverage"])
    result = {"scope": freeze["scope"], "fresh_holdout": False,
              "independent_confirmation": False,
              "checks_before_outcomes": checks, "failures": failures,
              "raw_confidence_reference": {"selected_error": raw_error,
                                           "achieved_coverage": raw_coverage},
              "metrics": metrics, "paired_bootstrap_vs_baseline": intervals,
              "model_diagnostics": diagnostics, "gates": verdicts,
              "wall_sec": time.perf_counter()-started,
              "freeze_sha256": digest(out/"freeze.json"),
              "decision": "calibration_complete" if not failures else "incomplete_inspect_failures"}
    write_json(out / "result.json", result)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        names = sorted(predictions)
        writer = csv.writer(f)
        writer.writerow(["case_id", "work_family", "performance_time_sec", "correct",
                         "confidence", "weight"] + names)
        for i, row in enumerate(rows):
            writer.writerow([row["case_id"], row["work_family"], row["performance_time_sec"],
                             int(y[i]), row["confidence"], weight[i]]
                            + [predictions[n][i] for n in names])
    write_json(out / "state.json", {"status": "COMPLETE" if not failures else "INCOMPLETE",
                                    "result_sha256": digest(out/"result.json"),
                                    "decision": result["decision"]})
    summary = {name: {"selected_error": m["selected_error_at_matched_coverage"],
                      "ece": m["ece"], "brier": m["brier"]} for name, m in metrics.items()}
    print(json.dumps({"raw_confidence": raw_error, "models": summary,
                      "gates": {k: v.get("pass", v) for k, v in (verdicts or {}).items()}},
                     indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
