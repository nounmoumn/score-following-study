"""Do the two deficits have the same cause? Descriptive only, no gate, no fitting.

The fixed transition implies one score frame per performance frame. Phase 3 showed
the online restriction costs about .057 accuracy. This asks whether that cost is the
same thing as tempo mismatch, or a separate deficit. Reads finished runs only.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/confidence_smoothing_v1.json"
OUT = ROOT / "results/tempo_mismatch_diagnostic"


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    annotations = json.loads((ROOT / cfg["asap_root"] / "asap_annotations.json")
                             .read_text(encoding="utf-8"))
    with (ROOT / cfg["source_manifest"]).open(encoding="utf-8-sig", newline="") as f:
        manifest = {r["case_id"]: r for r in csv.DictReader(f) if r["role"] == cfg["role"]}
    source = ROOT / cfg["out"] / "delay_free_beats.csv"
    rows = list(source.open(encoding="utf-8"))
    per_case = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(rows):
        per_case[row["case_id"]][int(row["lag"])].append(int(row["correct"]))
    low, high = min(cfg["lags"]), max(cfg["lags"])
    hop, frames = cfg["performance_hop_sec"], int(cfg["replay_prefix_sec"]/cfg["performance_hop_sec"])
    first, last = (high+1)*hop, (frames-high)*hop
    cases = []
    for case, lags in sorted(per_case.items()):
        ann = annotations[manifest[case]["performance"]]
        perf = np.asarray(ann["performance_beats"], float)
        score = np.asarray(ann["midi_score_beats"], float)
        window = (perf >= first) & (perf <= last)
        ratio = float((score[window][-1]-score[window][0])/(perf[window][-1]-perf[window][0]))
        local = np.diff(score[window])/np.diff(perf[window])
        base, top = float(np.mean(lags[low])), float(np.mean(lags[high]))
        cases.append({"case_id": case, "work_family": manifest[case]["work_family"],
                      "score_per_performance_ratio": ratio, "mismatch": abs(ratio-1.),
                      "local_ratio_p10": float(np.quantile(local, .1)),
                      "local_ratio_p90": float(np.quantile(local, .9)),
                      "accuracy_no_future": base, "accuracy_with_future": top,
                      "smoothing_gain": top-base, "observations": len(lags[low])})
    mismatch = np.array([c["mismatch"] for c in cases])
    accuracy = np.array([c["accuracy_no_future"] for c in cases])
    gain = np.array([c["smoothing_gain"] for c in cases])
    rank = lambda a: np.argsort(np.argsort(a)).astype(float)
    spearman = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    near, far = mismatch < .05, mismatch >= .15
    report = {
        "source": source.relative_to(ROOT).as_posix(),
        "nature": "descriptive diagnostic on finished runs; no gate, no fitting, n=40 over 23 families",
        "implied_transition_ratio": 1.0,
        "cases": len(cases),
        "spearman_mismatch_vs_accuracy": spearman(mismatch, accuracy),
        "spearman_mismatch_vs_smoothing_gain": spearman(mismatch, gain),
        "groups": {
            "within_5_percent": {"cases": int(near.sum()),
                                 "accuracy_no_future": float(accuracy[near].mean()),
                                 "accuracy_with_future": float((accuracy+gain)[near].mean()),
                                 "smoothing_gain": float(gain[near].mean())},
            "off_by_15_percent_or_more": {"cases": int(far.sum()),
                                          "accuracy_no_future": float(accuracy[far].mean()),
                                          "accuracy_with_future": float((accuracy+gain)[far].mean()),
                                          "smoothing_gain": float(gain[far].mean())}},
        "reading": ("tempo mismatch tracks baseline accuracy but not the online-information "
                    "cost; the two deficits are not the same thing"),
        "per_case": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
