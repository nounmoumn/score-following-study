"""Read-only validation of frozen pilot artifacts; does not rerun outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ambiguity_anchor_feasibility_v1"


def main():
    result = json.loads((OUT / "result.json").read_text())
    freeze = json.loads((OUT / "freeze.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("c[0-9][0-9][0-9].json"))]
    assert len(rows) == result["cases"] == 40
    works = sorted({r["work_family"] for r in rows})
    assert len(works) == result["works"] == 23
    assert sum(r["evaluation_rows"] for r in rows) == result["evaluation_rows"]
    assert all(r["evaluation_rows"] > 0 and len(r["conditions"]) == 16 for r in rows)
    for row in rows:
        for key, condition in row["conditions"].items():
            times = np.sort(condition["query_times"])
            assert len(times) == int(key.rsplit("k", 1)[1])
            assert np.all(np.diff(times) >= 5)
            assert np.all((times >= 15) & (times <= 85))
            assert np.all(times == times.astype(int))
            assert abs(condition["accuracy"] - row["baseline_accuracy"] -
                       condition["fixed_fraction"] + condition["broken_fraction"]) < 1e-12
    for key, metric in result["metrics"].items():
        equal_work_mean = np.mean([np.mean([r["conditions"][key]["accuracy"]
            for r in rows if r["work_family"] == work]) for work in works])
        assert abs(equal_work_mean - metric["accuracy"]) < 1e-12
    for path, digest in freeze["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    print(json.dumps({"checks": "PASS", "cases": len(rows), "works": len(works),
                      "conditions_per_case": 16, "input_hashes": len(freeze["files"]),
                      "evaluation_rows": result["evaluation_rows"]}, indent=2))


if __name__ == "__main__":
    main()
