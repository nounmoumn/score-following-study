import json
from pathlib import Path


def test_kraisler_authorization_uses_only_prespecified_risk_gates():
    root=Path(__file__).resolve().parents[1]
    result=json.loads((root/"results/p3_td_kraisler_transfer/result.json").read_text())
    required=("coverage","replay_and_manipulation","risk_curve_strictly_increasing",
              "positive_risk_rho_share_ge_0_80","ten_sec_risk_ge_0_25","ten_sec_risk_ci_lower_gt_0")
    assert all(result["gates"][key] for key in required)
    assert result["transfer_pass"] is False

