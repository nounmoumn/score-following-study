import numpy as np

from scripts import p3_sky_piano_temporal_transfer as transfer


def test_frozen_temporal_parameters_match_final_batik_test():
    assert transfer.WINDOW_SEC == 12.0
    assert transfer.HORIZON_SEC == transfer.LAG_SEC == transfer.STEP_SEC == 0.5
    assert transfer.RADIUS == 0.5211881394162992
    assert transfer.PERMUTATIONS == 2000


def test_summary_counts_onsets_and_tracks():
    rows = [
        {"track": "a", "event": True, "onset_sec": 1.0, "deficit": 2.0},
        {"track": "a", "event": True, "onset_sec": 1.0, "deficit": 1.0},
        {"track": "b", "event": False, "onset_sec": None, "deficit": 0.0},
    ]
    result = transfer.summarize(rows, np.random.default_rng(1))
    assert result["counts"]["samples"] == 2
    assert result["counts"]["unique_onsets"] == 1
