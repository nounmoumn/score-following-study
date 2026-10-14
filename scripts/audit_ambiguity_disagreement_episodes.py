"""Independent saved-output audit for disagreement episode analysis."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "ambiguity_disagreement_episodes"
OUT = SOURCE / "independent_audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    result = json.loads((SOURCE / "result.json").read_text(encoding="utf-8"))
    episodes = json.loads((SOURCE / "episodes.json").read_text(encoding="utf-8"))
    freeze = json.loads((SOURCE / "freeze.json").read_text(encoding="utf-8"))
    drift = [relative for relative, expected in freeze.items() if digest(ROOT / relative) != expected]
    records = result["detector"]["per_work"]
    comparable = [record for record in records if "selector_error_flagged" in record]
    flagged_beats = sum(record["flagged_beats"] for record in records)
    flagged_works = sum(record["flagged_beats"] > 0 for record in records)
    coverage = float(np.mean([record["coverage"] for record in records]))
    error_flagged = float(np.mean([record["selector_error_flagged"] for record in comparable]))
    error_unflagged = float(np.mean([record["selector_error_unflagged"] for record in comparable]))
    regret_flagged = float(np.mean([record["high_regret_flagged"] for record in comparable]))
    regret_unflagged = float(np.mean([record["high_regret_unflagged"] for record in comparable]))
    positive = sum(record["selector_error_flagged"] > record["selector_error_unflagged"] for record in comparable)
    required = max(3, int(np.ceil(0.6*len(comparable))))
    gates = {
        "support": flagged_beats >= 20 and flagged_works >= 3 and coverage >= .05,
        "selector_error_ratio": error_flagged/max(error_unflagged, 1e-12) >= 1.5,
        "high_regret_ratio": regret_flagged/max(regret_unflagged, 1e-12) >= 1.5,
        "positive_work_fraction": len(comparable) >= 3 and positive >= required,
    }
    gates["pass"] = all(gates.values())
    category_violations = []
    for episode in episodes:
        similarity = episode["median_direct_similarity"]
        if episode["start_sec"] <= 10:
            expected = "initialization"
        elif episode["max_tracker_jump"] >= 2:
            expected = "relocation"
        elif similarity is not None and similarity >= .8 and episode["median_score_separation"] >= 2:
            expected = "structural_rival"
        elif episode["median_speed_difference"] >= .5:
            expected = "tempo_divergence"
        else:
            expected = "residual"
        if expected != episode["category"]:
            category_violations.append(episode["episode_id"])
    numeric_residuals = [
        abs(flagged_beats-result["detector"]["flagged_beats"]),
        abs(flagged_works-result["detector"]["flagged_works"]),
        abs(coverage-result["detector"]["work_equal_disagreement_coverage"]),
        abs(error_flagged-result["detector"]["selector_error"]["flagged"]),
        abs(error_unflagged-result["detector"]["selector_error"]["unflagged"]),
        abs(regret_flagged-result["detector"]["high_regret"]["flagged"]),
        abs(regret_unflagged-result["detector"]["high_regret"]["unflagged"]),
    ]
    category_counts = dict(sorted(Counter(episode["category"] for episode in episodes).items()))
    audit = {
        "status": "PASS" if not drift and not category_violations and max(numeric_residuals) <= 1e-12 and gates == result["gates"] else "FAIL",
        "episodes": len(episodes),
        "category_counts": category_counts,
        "category_violations": category_violations,
        "max_numeric_residual": max(numeric_residuals),
        "gates_recomputed": gates,
        "hash_drift": drift,
        "lockbox_output_absent": not (ROOT / "results" / "ambiguity_disagreement_lockbox").exists(),
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
