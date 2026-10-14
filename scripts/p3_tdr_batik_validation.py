#!/usr/bin/env python3
"""One-shot Batik validation of the continuous P3-TDR duration endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
import p3_td_transfer as transfer


def main():
    prior=json.loads((ROOT/"results/p3_td_kraisler_transfer/result.json").read_text(encoding="utf8"))
    required=("coverage","replay_and_manipulation","risk_curve_strictly_increasing",
              "positive_risk_rho_share_ge_0_80","ten_sec_risk_ge_0_25","ten_sec_risk_ci_lower_gt_0")
    if not all(prior["gates"][key] for key in required):
        raise SystemExit("KRAISLER risk endpoint did not authorize Batik validation")
    import p3_pt_batik_heterogeneity as batik
    manifest=json.loads((ROOT/"results/p3_batik_external/manifest.json").read_text(encoding="utf8"))
    cases=[dict(row,unit=f"{row['sonata']}-{row['movement']}") for row in manifest["rows"] if row["eligible"]]
    def loader(case,hop):
        u,v,score_hop,truth=batik.load_case(case,hop)
        return u,v,score_hop,truth,min(90.,float(case["duration_sec"]))
    result=transfer.run_cases(cases,loader,"unit",
        lambda case,available:np.arange(15.,available-10.4+1e-9,10.),
        ROOT/"results/p3_tdr_batik_validation","Batik one-shot P3-TDR validation",
        lambda units,landmarks:units>=15 and landmarks>=90)
    risk_gates={key:result["gates"][key] for key in required}
    result["p3_tdr_risk_gates"]=risk_gates
    result["p3_tdr_confirmed"]=bool(all(risk_gates.values()))
    result["incidence_is_descriptive"]=True
    result["original_natural_p3_confirmed"]=False
    (ROOT/"results/p3_tdr_batik_validation/result.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps({"p3_tdr_risk_gates":risk_gates,"p3_tdr_confirmed":result["p3_tdr_confirmed"]},indent=2))


if __name__=="__main__":main()

