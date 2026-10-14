import pytest

from scripts import p3_fhvb_smoothness_moderator as moderator


def test_track_roughness_does_not_use_event_labels():
    base = [
        {"track": "a", "time_sec": 0.0, "deficit": 0.1, "event": False},
        {"track": "a", "time_sec": 0.5, "deficit": 0.2, "event": True},
        {"track": "a", "time_sec": 1.0, "deficit": 0.4, "event": False},
    ]
    flipped = [{**row, "event": not row["event"]} for row in base]
    first = moderator.track_records({"Batik-development": base})[0]
    second = moderator.track_records({"Batik-development": flipped})[0]
    assert first["background_roughness"] == second["background_roughness"]
    assert first["background_roughness"] == pytest.approx(0.15)


def test_smooth_threshold_is_fixed():
    assert moderator.SMOOTH_THRESHOLD == 0.03
    assert moderator.PERMUTATIONS == 20_000
