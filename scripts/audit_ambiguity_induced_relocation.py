"""Independent saved-output audit for the ambiguity-induced relocation test."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"results"/"ambiguity_induced_relocation"
OUT = SOURCE/"independent_audit.json"
SEED = 20260924


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    sampled = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return np.quantile(sampled.mean(axis=1), [.025, .975]).tolist()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refuse overwrite: {OUT}")
    result = json.loads((SOURCE/"result.json").read_text(encoding="utf-8"))
    events = json.loads((SOURCE/"events.json").read_text(encoding="utf-8"))
    freeze = json.loads((SOURCE/"freeze.json").read_text(encoding="utf-8"))
    drift = [relative for relative, expected in freeze.items() if digest(ROOT/relative) != expected]
    by_work = defaultdict(list)
    for event in events:
        by_work[event["work"]].append(event)
    differences = []
    event_rates = []
    control_rates = []
    for work, rows in sorted(by_work.items()):
        differences.append(np.mean([row["origin_maxsim"]-row["control_maxsim"] for row in rows]))
        event_rates.append(np.mean([row["origin_maxsim"] >= .8 for row in rows]))
        control_rates.append(np.mean([row["control_maxsim"] >= .8 for row in rows]))
    differences = np.asarray(differences, float)
    percentiles = np.asarray([event["destination_percentile"] for event in events], float)
    event_high, control_high = float(np.mean(event_rates)), float(np.mean(control_rates))
    methods = sorted(set(event["method"] for event in events))
    positive = int(np.sum(differences > 0))
    gates = {
        "support": bool(len(events) >= 20 and len(by_work) >= 4 and methods == ["arzt", "outerhmm"]),
        "destination_rival": bool(np.median(percentiles) >= .90 and np.mean(percentiles >= .95) >= .50),
        "precursor_difference": bool(differences.mean() > 0 and ci(differences)[0] > 0 and positive >= int(np.ceil(.6*len(differences)))),
        "high_ambiguity_ratio": bool(event_high/max(control_high, 1e-12) >= 1.5),
    }
    gates["pass"] = all(gates.values())
    residuals = [
        abs(len(events)-result["matched_events"]),
        abs(len(by_work)-result["works"]),
        abs(float(np.median(percentiles))-result["destination"]["median_percentile"]),
        abs(float(np.mean(percentiles >= .95))-result["destination"]["percentile_95_fraction"]),
        abs(float(differences.mean())-result["precursor"]["work_equal_mean_maxsim_difference"]),
        abs(event_high-result["precursor"]["event_high_ambiguity_rate"]),
        abs(control_high-result["precursor"]["control_high_ambiguity_rate"]),
    ]
    audit = {
        "status": "PASS" if not drift and max(residuals) <= 1e-12 and gates == result["gates"] else "FAIL",
        "matched_events": len(events),
        "works": len(by_work),
        "methods": methods,
        "outcomes": dict(sorted(Counter(event["outcome"] for event in events).items())),
        "max_numeric_residual": max(residuals),
        "bootstrap95_recomputed": ci(differences),
        "gates_recomputed": gates,
        "hash_drift": drift,
        "lockbox_output_absent": not (ROOT/"results"/"ambiguity_induced_relocation_lockbox").exists(),
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
