#!/usr/bin/env python3
"""Post-failure point-centered conformal diagnostic; no qualification gate."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"results/p3_vienna_forecast_validation/landmarks.csv"
OUT=ROOT/"results/p3_vienna_point_interval_diagnostic"

def higher(values,q):return float(np.percentile(np.asarray(values,dtype=float),q,method="higher"))

def main():
    with SOURCE.open(encoding="utf8",newline="") as h:rows=list(csv.DictReader(h))
    for row in rows:
        row["center_error_sec"]=abs(float(row["center_state"])-float(row["truth_state"]))*float(row["score_hop"])
    calibration=[r for r in rows if r["split"]=="calibration"]
    stems=sorted({r["stem"] for r in calibration})
    perf_q={stem:higher([r["center_error_sec"] for r in calibration if r["stem"]==stem],95) for stem in stems}
    radius=higher(list(perf_q.values()),95)
    validation=[r for r in rows if r["split"]=="validation"]
    coverage={work:float(np.mean([r["center_error_sec"]<=radius for r in validation if r["work"]==work]))
              for work in sorted({r["work"] for r in validation})}
    result={"scope":"post-failure diagnostic on opened Vienna validation; cannot qualify a model",
            "performance_cluster_radius_sec":radius,"validation_work_coverage":coverage,
            "work_equal_coverage":float(np.mean(list(coverage.values()))),
            "candidate_for_fresh_cohort":bool(radius<=.75 and min(coverage.values())>=.85),
            "natural_p3_confirmed":False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
