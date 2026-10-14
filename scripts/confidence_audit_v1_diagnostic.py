"""Post-hoc descriptive diagnostic on the frozen v1 audit. NOT the calibration study.

Runs only on results/confidence_audit_v1/primary_beats.csv, fits nothing, optimizes
no threshold, and writes to a separate directory. Its purpose is the go/no-go read on
whether A (score ambiguity) and e (observation support) carry any correctness signal
beyond ordinary GT-free confidence. Effect sizes here are unweighted-by-design inside
strata and must not be reported as the calibration result.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/confidence_audit_v1/primary_beats.csv"
OUT = ROOT / "results/confidence_audit_v1_diagnostic"


def load():
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    col = lambda k: np.array([float(r[k]) for r in rows])
    case = np.array([r["case_id"] for r in rows])
    work = np.array([r["work_family"] for r in rows])
    counts = Counter(zip(work, case))
    families = defaultdict(set)
    for wf, ci in counts:
        families[wf].add(ci)
    w = np.array([1./len(families)/len(families[wf])/counts[(wf, ci)]
                  for wf, ci in zip(work, case)])
    return dict(y=col("correct"), c=col("confidence"), e=col("evidence"),
                cprior=col("prior_confidence"),
                A=col("ambiguity"), entropy=col("entropy"), margin=col("state_margin"),
                pmax=col("pmax"), mass=col("ambiguity_valid_mass"),
                case=case, work=work, weight=w/w.sum())


def tercile_split(d, base, key):
    """Accuracy in the outer terciles of `key`, weighted by the audit's work weights."""
    v = d[key][base]
    q = np.quantile(v, [1/3, 2/3])
    out = {}
    for name, mask in (("low", v <= q[0]), ("high", v >= q[1])):
        sel = base.copy()
        sel[base] = mask
        ww = d["weight"][sel]
        out[name] = {"accuracy": float(ww @ d["y"][sel]/ww.sum()), "observations": int(sel.sum())}
    out["spread"] = out["high"]["accuracy"] - out["low"]["accuracy"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    high = d["c"] >= .9
    base = high & (d["mass"] >= .99)
    report = {"source": SOURCE.relative_to(ROOT).as_posix(),
              "nature": "descriptive diagnostic, no fitted model, no threshold search",
              "high_confidence_observations": int(high.sum()),
              "base_observations_defined_ambiguity": int(base.sum())}

    # e == 0 is NOT numerical clipping. cosine_cost returns a constant row when the
    # performance chroma frame has zero energy, so the likelihood is genuinely flat.
    # Verified on c004: e == 0 agreed with zero-energy frames on 65/65 beats.
    # Clipping never fired on this cohort: min(1-c) = 2.6e-9, min(1-u) = 1.0e-6.
    silence = d["e"] == 0
    report["evidence_scale"] = {
        "flat_likelihood_fraction": float(np.mean(silence)),
        "flat_likelihood_is_clipping": False,
        "min_one_minus_confidence": float((1-d["c"]).min()),
        "min_one_minus_prior_confidence": float((1-d["cprior"]).min()),
        "observations_with_either_mass_clipped_at_1e-12": int(
            np.sum((1-d["c"] <= 1e-12) | (1-d["cprior"] <= 1e-12))),
        "confidence_above_1_minus_1e6_fraction": float(np.mean(d["c"] > 1-1e-6)),
        "confidence_above_0999_fraction": float(np.mean(d["c"] > .999))}

    # The tracker is most overconfident exactly where it receives no usable
    # observation. A flat likelihood row is detectable at runtime without GT.
    def weighted(mask):
        ww = d["weight"][mask]
        return {"accuracy": float(ww @ d["y"][mask]/ww.sum()),
                "mean_confidence": float(ww @ d["c"][mask]/ww.sum()),
                "weight": float(ww.sum()), "observations": int(mask.sum())}
    strata = {"flat_likelihood": weighted(silence), "observed": weighted(~silence),
              "flat_likelihood_high_confidence": weighted(silence & high),
              "observed_high_confidence": weighted(~silence & high)}
    for key, value in strata.items():
        value["overconfidence_gap"] = value["mean_confidence"] - value["accuracy"]
    err = {k: float(d["weight"][m] @ (1-d["y"][m]))
           for k, m in (("flat_likelihood", silence & high), ("observed", ~silence & high))}
    strata["high_confidence_error_mass"] = err
    strata["high_confidence_error_share_from_flat_likelihood"] = (
        err["flat_likelihood"]/sum(err.values()))
    report["flat_likelihood_strata"] = strata

    # Confound check: is e's separation merely silence versus observed?
    report["e_terciles_observed_frames_only"] = tercile_split(d, high & ~silence, "e")

    report["marginal_terciles_within_high_confidence"] = {
        key: tercile_split(d, base if key == "A" else high, key)
        for key in ("A", "e", "entropy", "margin", "pmax")}

    # Conditional read: hold an ordinary GT-free confidence proxy fixed, then split.
    q = np.quantile(d["entropy"][base], [1/3, 2/3])
    conditional = {}
    for key in ("A", "e"):
        strata = []
        for i, (lo, hi) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
            stratum = base & (d["entropy"] > lo) & (d["entropy"] <= hi)
            strata.append({"entropy_tercile": i+1, **tercile_split(d, stratum, key)})
        conditional[key] = strata
    report["conditional_on_entropy_tercile"] = conditional

    # Between- vs within-performance decomposition: a pooled effect that exists only
    # between performances is closer to 'some pieces are harder' than to a per-frame signal.
    cases = sorted(set(d["case"]))
    means = [(d["A"][base & (d["case"] == ci)].mean(), d["y"][base & (d["case"] == ci)].mean())
             for ci in cases if (base & (d["case"] == ci)).sum() >= 20]
    x, z = np.array([m[0] for m in means]), np.array([m[1] for m in means])
    rank = lambda a: np.argsort(np.argsort(a)).astype(float)
    decomposition = {"between_performance": {
        "cases": len(means), "spearman_meanA_accuracy": float(np.corrcoef(rank(x), rank(z))[0, 1])}}
    for key in ("A", "e"):
        deltas = []
        for ci in cases:
            m = base & (d["case"] == ci)
            if m.sum() < 40 or d["y"][m].std() == 0:
                continue
            qq = np.quantile(d[key][m], [1/3, 2/3])
            if qq[0] == qq[1]:
                continue
            deltas.append(float(d["y"][m & (d[key] >= qq[1])].mean()
                                - d["y"][m & (d[key] <= qq[0])].mean()))
        deltas = np.array(deltas)
        decomposition[f"within_performance_{key}"] = {
            "cases": len(deltas), "median_delta": float(np.median(deltas)),
            "mean_delta": float(deltas.mean()),
            "cases_in_expected_direction": int(np.sum(deltas < 0) if key == "A" else np.sum(deltas > 0))}
    report["between_vs_within_performance"] = decomposition
    (OUT / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
