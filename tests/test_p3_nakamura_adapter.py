from pathlib import Path

import numpy as np

from scripts import p3_nakamura_adapter as adapter


def test_parse_corresp_and_clock(tmp_path: Path):
    path = tmp_path / "test_corresp.txt"
    path.write_text(
        "// header\n"
        "0\t1.0\tC4\t60\t70\t2\t0.5\tC4\t60\t80\t\n"
        "1\t1.0\tE4\t64\t70\t3\t0.75\tE4\t64\t80\t\n"
        "2\t2.0\tG4\t67\t70\t4\t1.0\tG4\t67\t80\t\n"
        "*\t-1\t*\t-1\t-1\t5\t1.5\tA4\t69\t80\t\n",
        encoding="utf8",
    )
    matched, missing, extra = adapter.parse_corresp(path)
    anchors = adapter.clock_anchors(matched)
    assert len(matched) == 3 and len(missing) == 1 and len(extra) == 0
    assert np.allclose(anchors, [[1.0, .625], [2.0, 1.0]])
    assert adapter.audit_corresp(path)["clock_monotone"]


def test_windows_to_wsl_path_shape():
    converted = adapter.windows_to_wsl(adapter.ROOT / "results")
    assert converted.startswith("/mnt/")
    assert converted.endswith("/results")
