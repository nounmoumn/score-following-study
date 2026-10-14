"""Post-hoc descriptive diagnostic on the frozen phase-2 run. Not a gate.

Tests the report's explanation directly: if calibration fixes the probability
scale but leaves the ORDER of outputs alone, then discrimination (which depends
only on order) should be nearly identical across every model, while calibration
metrics move a lot. Fits nothing; separate output directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/confidence_calibration_v1/predictions.csv"
OUT = ROOT / "results/confidence_calibration_v1_diagnostic"


def weighted_auc(score, y, w):
    """Weighted AUC = P(score of a correct row > score of a wrong row), ties at .5."""
    order = np.argsort(score, kind="stable")
    s, y, w = score[order], y[order], w[order]
    pos = w*y
    neg = w*(1-y)
    total = 0.
    seen_neg = 0.
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        tie_pos, tie_neg = pos[i:j].sum(), neg[i:j].sum()
        total += tie_pos*(seen_neg + tie_neg/2)
        seen_neg += tie_neg
        i = j
    denominator = pos.sum()*neg.sum()
    return float(total/denominator) if denominator else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    y = np.array([float(r["correct"]) for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    raw = np.array([float(r["confidence"]) for r in rows])
    models = [k for k in rows[0] if k not in
              {"case_id", "work_family", "performance_time_sec", "correct", "confidence", "weight"}]
    report = {"source": SOURCE.relative_to(ROOT).as_posix(),
              "nature": "descriptive diagnostic on frozen predictions; no gate, no fitting",
              "observations": len(rows)}
    auc = {"raw_confidence": weighted_auc(raw, y, w)}
    spearman = {}
    rank = lambda a: np.argsort(np.argsort(a)).astype(float)
    for name in models:
        p = np.array([float(r[name]) for r in rows])
        auc[name] = weighted_auc(p, y, w)
        spearman[name] = float(np.corrcoef(rank(p), rank(raw))[0, 1])
    report["weighted_auc"] = auc
    report["auc_spread_across_models"] = max(auc.values()) - min(auc.values())
    report["rank_correlation_with_raw_confidence"] = spearman
    # Ceiling check: a single monotone recalibration of raw confidence cannot change AUC.
    report["monotone_recalibration_cannot_change_order"] = {
        "raw_auc": auc["raw_confidence"],
        "baseline_calibrator_auc": auc.get("B"),
        "difference": None if auc.get("B") is None else auc["B"] - auc["raw_confidence"]}

    # Stability: a pooled AUC driven by a few works would not support the claim that
    # confidence is well ordered. Within each family every observation weighs equally.
    family = np.array([r["work_family"] for r in rows])
    baseline = np.array([float(r["B"]) for r in rows]) if "B" in rows[0] else None
    per_family, single_class = [], []
    for name in sorted(set(family)):
        mask = family == name
        uniform = np.full(int(mask.sum()), 1./int(mask.sum()))
        value = weighted_auc(raw[mask], y[mask], uniform)
        if value is None:
            single_class.append(name)
            continue
        entry = {"work_family": name, "observations": int(mask.sum()),
                 "accuracy": float(y[mask].mean()), "raw_auc": value}
        if baseline is not None:
            entry["baseline_calibrator_auc"] = weighted_auc(baseline[mask], y[mask], uniform)
        per_family.append(entry)
    values = np.array([e["raw_auc"] for e in per_family])
    report["per_work_family_raw_auc"] = {
        "families_scored": len(per_family), "families_single_class": single_class,
        "median": float(np.median(values)), "min": float(values.min()),
        "max": float(values.max()), "below_one_half": int((values < .5).sum()),
        "raw_better_than_baseline_calibrator": int(sum(
            1 for e in per_family if e.get("baseline_calibrator_auc") is not None
            and e["raw_auc"] > e["baseline_calibrator_auc"])),
        "families": per_family}
    (OUT / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
