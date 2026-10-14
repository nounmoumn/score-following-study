"""Post-hoc robustness diagnostics; does not change the frozen primary family."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from score_following.prospective import bootstrap_mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    args = p.parse_args()
    out = ROOT/args.result
    with open(out/"composition_effects.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for tracker, comparison in [("shared", "A_to_weakness"), ("bayes", "recovery_vs_target"),
                                ("oltw", "prevention_vs_target"), ("oltw", "recovery_vs_target")]:
        rr = [r for r in rows if r["cohort"]=="holdout" and r["tracker"]==tracker and r["comparison"]==comparison]
        x = np.array([float(r["effect"]) for r in rr])
        composers = sorted({r["composition"].split('/')[0] for r in rr})
        cluster = np.array([np.mean([float(r["effect"]) for r in rr if r["composition"].split('/')[0]==c]) for c in composers])
        largest = int(np.argmax(x))
        without = np.delete(x, largest)
        result[tracker+"_"+comparison] = dict(mean=float(x.mean()), median=float(np.median(x)),
            positive_over_1e8=int(np.sum(x>1e-8)), zero_within_1e8=int(np.sum(abs(x)<=1e-8)),
            max=float(x[largest]), largest_composition=rr[largest]["composition"],
            max_share_of_sum=float(x[largest]/x.sum()) if x.sum() else None,
            without_largest_mean=float(without.mean()),
            without_largest_ci95=bootstrap_mean(without, 20260918),
            composers=len(composers), composer_balanced_mean=float(cluster.mean()),
            composer_bootstrap_ci95=bootstrap_mean(cluster, 20260918))
    (out/"posthoc_robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
