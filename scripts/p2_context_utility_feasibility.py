"""One bounded, post-result feasibility diagnostic. No fitting or policy search."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.confidence_audit import blocked_maxsim
from score_following.features import load_notes, notes_to_chroma_stream


def read(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT / "results/p2_context_utility_feasibility"
    if out.exists():
        raise SystemExit("Refusing overwrite")
    manifest = ROOT / "output/score_priority_plan/cohort_manifest.csv"
    cases = {r["case_id"]: r for r in read(manifest) if r["role"] == "primary_opened"}
    paths = [Path(__file__), ROOT / "P2_CONTEXT_UTILITY_FEASIBILITY_PROTOCOL.md", manifest,
             ROOT / "results/confidence_audit_v2/primary_beats.csv",
             ROOT / "results/confidence_smoothing_v1/delay_free_beats.csv",
             ROOT / "src/score_following/confidence_audit.py",
             ROOT / "src/score_following/features.py"]
    paths += sorted({ROOT / "asap_dataset" / r["score"] for r in cases.values()})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out.mkdir()
    (out / "freeze.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    audit = {(r["case_id"], r["performance_time_sec"]): r for r in read(paths[3])}
    smooth = read(paths[4])
    by_lag = {lag: {(r["case_id"], r["performance_time_sec"]): r for r in smooth if int(r["lag"]) == lag}
              for lag in (0, 5, 10, 20)}
    assert all(set(v) == set(by_lag[0]) for v in by_lag.values())
    cache = {}
    with threadpool_limits(limits=2):
        for case in cases.values():
            path = case["score"]
            if path in cache:
                continue
            notes = load_notes(ROOT / "asap_dataset" / path)
            chroma = notes_to_chroma_stream(notes, max(n[1] for n in notes), .1)
            cache[path] = blocked_maxsim(chroma, 20, 20, 256)[0]
    observations = []
    for key, base in by_lag[0].items():
        a = audit[key]
        position = float(base["truth_score_sec"])
        idx = int(round(position / .1)) - 1
        profile = cache[cases[key[0]]["score"]]
        true = float(profile[idx]) if 0 <= idx < len(profile) else np.nan
        online = float(a["ambiguity"]) if float(a["ambiguity_valid_mass"]) >= .99 else np.nan
        row = {"case_id": key[0], "work_family": base["work_family"], "time": key[1],
               "A_true": true, "A_online": online, "confidence": float(a["confidence"]),
               "GT_grid_error": abs((idx+1)*.1-position), "base_correct": int(base["correct"])}
        for lag in (5, 10, 20):
            row[f"gain_{lag}"] = int(by_lag[lag][key]["correct"]) - int(base["correct"])
        observations.append(row)
    result = {"scope": "opened development feasibility only", "rows": len(observations),
              "max_GT_grid_error_score_sec": max(r["GT_grid_error"] for r in observations), "features": {}}
    for feature in ("A_true", "A_online", "confidence"):
        per_case = []
        missing = 0
        for case_id in cases:
            sample = [r for r in observations if r["case_id"] == case_id and np.isfinite(r[feature])]
            missing += sum(r["case_id"] == case_id for r in observations) - len(sample)
            if not sample:
                continue
            x, y = np.array([r[feature] for r in sample]), np.array([r["gain_20"] for r in sample])
            sd = x.std()
            per_case.append({"case_id": case_id, "work_family": cases[case_id]["work_family"],
                "n": len(sample), "mean_A": float(x.mean()), "mean_gain": float(y.mean()),
                "slope_per_within_case_SD": float(((x-x.mean())/sd) @ y / len(y)) if sd > 1e-12 else None})
        works = {}
        for row in per_case:
            works.setdefault(row["work_family"], []).append(row)
        slope = [np.mean([r["slope_per_within_case_SD"] for r in rs if r["slope_per_within_case_SD"] is not None])
                 for rs in works.values() if any(r["slope_per_within_case_SD"] is not None for r in rs)]
        slope = np.array(slope)
        rng = np.random.default_rng(20260923)
        draws = slope[rng.integers(0, len(slope), (10000, len(slope)))].mean(1)
        result["features"][feature] = {"missing_rows": missing, "works": len(slope),
            "mean_within_case_slope": float(slope.mean()),
            "bootstrap95": np.quantile(draws, [.025, .975]).tolist(),
            "positive_works": int((slope>0).sum()), "negative_works": int((slope<0).sum()),
            "between_work_spearman": float(spearmanr([np.mean([r["mean_A"] for r in rs]) for rs in works.values()],
                                                       [np.mean([r["mean_gain"] for r in rs]) for rs in works.values()]).statistic),
            "per_case": per_case}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    with (out / "observations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({**result, "features": {k: {a:b for a,b in v.items() if a != "per_case"}
                                            for k,v in result["features"].items()}}, indent=2))


if __name__ == "__main__":
    main()
