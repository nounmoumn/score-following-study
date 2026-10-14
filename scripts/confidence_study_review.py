"""Read-only post-result audit; no fitting, feature search, or gate changes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT / "results/confidence_study_review"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    result = {"scope": "post-result diagnostic, not confirmation", "hash_checks": {}}
    for name in ("confidence_audit_v1", "confidence_audit_v2",
                 "confidence_calibration_v1", "confidence_smoothing_v1"):
        freeze = json.loads((ROOT / "results" / name / "freeze.json").read_text())
        changed = [p for p, h in freeze["files"].items()
                   if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
        result["hash_checks"][name] = {"files": len(freeze["files"]), "changed": changed}
    audit = rows(ROOT / "results/confidence_audit_v2/primary_beats.csv")
    smooth = rows(ROOT / "results/confidence_smoothing_v1/delay_free_beats.csv")
    lookup = {(r["case_id"], r["performance_time_sec"]): r for r in audit}
    base = [r for r in smooth if r["lag"] == "0"]
    result["lag0_equivalence"] = {
        "rows": len(base),
        "position_mismatches": sum(float(r["position_score_sec"]) != float(lookup[(r["case_id"], r["performance_time_sec"])]["position_score_sec"]) for r in base),
        "label_mismatches": sum(r["correct"] != lookup[(r["case_id"], r["performance_time_sec"])]["correct"] for r in base),
    }
    # Cluster means, then paired bootstrap works. Conditions share exact anchors.
    by_lag = {}
    for lag in ("0", "5", "10", "20"):
        cases = {}
        for r in smooth:
            if r["lag"] == lag:
                cases.setdefault((r["work_family"], r["case_id"]), []).append(float(r["correct"]))
        works = {}
        for (work, _), values in cases.items():
            works.setdefault(work, []).append(np.mean(values))
        by_lag[lag] = {w: float(np.mean(v)) for w, v in works.items()}
    families = sorted(by_lag["0"])
    delta = np.array([by_lag["20"][w] - by_lag["0"][w] for w in families])
    rng = np.random.default_rng(20260923)
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    result["retrospective_accuracy_gain"] = {
        "work_equal": float(delta.mean()), "positive_works": int((delta > 0).sum()),
        "negative_works": int((delta < 0).sum()), "works": len(delta),
        "posthoc_work_bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}
    # Even PERFECT past positions become stale when scored against current truth.
    # Uses interpolated annotations and is an oracle illustration, not a live method.
    annotations = json.loads((ROOT / "asap_dataset/asap_annotations.json").read_text())
    manifest = rows(ROOT / "output/score_priority_plan/cohort_manifest.csv")
    paths = {r["case_id"]: r["performance"] for r in manifest}
    stale = {}
    for lag in (0, 5, 10, 20):
        cases = {}
        for r in base:
            ann = annotations[paths[r["case_id"]]]
            t = float(r["performance_time_sec"])
            past = (np.floor(t / .1) - lag) * .1
            times = ann["performance_beats"]
            if past < times[0] or past > times[-1]:
                continue
            truth_past = np.interp(past, times, ann["midi_score_beats"])
            correct = abs(truth_past - float(r["truth_score_sec"])) <= .5 + 1e-12
            cases.setdefault((r["work_family"], r["case_id"]), []).append(correct)
        works = {}
        for (work, _), values in cases.items():
            works.setdefault(work, []).append(np.mean(values))
        stale[str(lag)] = {"work_equal_accuracy": float(np.mean([np.mean(v) for v in works.values()])),
                           "rows": sum(map(len, cases.values()))}
    result["perfect_past_position_displayed_as_current_oracle"] = stale
    result["inputs_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "results/confidence_audit_v2/primary_beats.csv",
                  ROOT / "results/confidence_smoothing_v1/delay_free_beats.csv",
                  ROOT / "asap_dataset/asap_annotations.json",
                  ROOT / "output/score_priority_plan/cohort_manifest.csv"]}
    out.mkdir()
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
