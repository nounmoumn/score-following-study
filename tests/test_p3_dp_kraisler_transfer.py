import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_dp_kraisler_transfer.py"
SPEC = importlib.util.spec_from_file_location("p3_dp_kraisler_transfer", SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def test_aggregate_keeps_map_as_diagnostic_and_posterior_as_primary():
    rows = []
    for track in ("a", "b"):
        for value in (1.0, .75, .5, .25, 0.0):
            for window in range(2):
                rows.append({"track": track, "lambda": value,
                             "posterior_js": (1-value) + window*.01,
                             "map_divergence_sec": 0.0})
    doses, effects = transfer.aggregate(rows)
    assert len(doses) == 10 and len(effects) == 2
    assert all(row["posterior_dose_rho"] > .99 for row in effects)
    assert all(row["map_dose_rho_diagnostic"] != row["map_dose_rho_diagnostic"] for row in effects)

