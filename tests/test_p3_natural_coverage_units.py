from scripts.p3_natural_coverage_units import unit_metrics


def _events(count):
    return [{"time_sec": float(i), "latest_end_sec": float(i)+.5} for i in range(count)]


def test_monotone_wide_unit_passes():
    anchors = [{"performance_index": i, "score_index": i} for i in range(100)]
    result = unit_metrics(anchors, _events(100), 100, 0, 100)
    assert result["reviewable"]


def test_narrow_or_backward_unit_fails():
    narrow = [{"performance_index": i, "score_index": 40+i//10} for i in range(100)]
    assert not unit_metrics(narrow, _events(100), 100, 0, 100)["reviewable"]
    backward = [{"performance_index": i, "score_index": 99-i} for i in range(100)]
    assert not unit_metrics(backward, _events(100), 100, 0, 100)["reviewable"]
