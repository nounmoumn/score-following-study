import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_td_duration_response.py"
SPEC = importlib.util.spec_from_file_location("p3_td_duration_response", SCRIPT)
duration = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(duration)


def test_aggregate_preserves_duration_order_and_unit_weighting():
    rows=[]
    for unit in ("a","b"):
        for landmark in range(2):
            for seconds in duration.DURATIONS:
                rows.append({"unit":unit,"duration_sec":seconds,"risk_effect":seconds+landmark,
                             "erased_event":int(seconds==10),"intact_event":0})
    doses,effects=duration.aggregate(rows,"unit")
    assert len(doses)==6 and len(effects)==2
    assert all(row["risk_duration_rho"]==1 for row in effects)

