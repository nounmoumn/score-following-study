#!/usr/bin/env python3
"""Build reviewer-locked, mutually isolated phase-1 review kits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/p3_natural_boundary_review_packet"
OUT = ROOT / "results/p3_natural_review_kits"
REVIEWERS = ("reviewer_A", "reviewer_B")
ALLOWED_CANDIDATE_KEYS = {
    "candidate_id", "record_time", "composer", "piece", "time_sec", "signals",
    "context_top_pitches", "decision", "reviewer", "reason", "clip_media", "overview",
}
ALLOWED_FILE_KEYS = {
    "record_time", "composer", "piece", "performance_path", "score_path",
    "performance_sha256", "score_sha256", "candidate_count", "reviewable_units",
    "missed_boundary_check", "reviewer", "review_reason", "performance_media",
    "score_media", "duration_sec", "overview",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded_data(page):
    return json.loads(page.split("const DATA=", 1)[1].split(";\nconst state", 1)[0])


def customize_html(page, reviewer):
    label = "A" if reviewer == "reviewer_A" else "B"
    other = "reviewer_B" if reviewer == "reviewer_A" else "reviewer_A"
    page = re.sub(
        r'<label>Reviewer <select id="reviewer">.*?</select></label>',
        f'<span>Reviewer {label} (locked)</span><input id="reviewer" type="hidden" value="{reviewer}">',
        page,
        count=1,
    )
    page = page.replace('<button data-tab="units">Provisional units</button>', "")
    page = re.sub(r',"units":\[.*?\],"settings":', ',"units":[],"settings":', page, count=1, flags=re.S)
    page = page.replace("reviewer:'reviewer_A'", f"reviewer:'{reviewer}'", 1)
    page = page.replace("p3-natural-review", f"p3-natural-review-{label}")
    if other in page:
        raise RuntimeError(f"Reviewer isolation failed: {other} remains in {reviewer} HTML")
    return page


def copy_template(source, destination, reviewer):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames
    for row in rows:
        row["reviewer"] = reviewer
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_payload(page, reviewer):
    data = embedded_data(page)
    assert set(data) == {"candidates", "files", "units", "settings"}
    assert data["units"] == []
    assert len(data["candidates"]) == 41 and len(data["files"]) == 29
    assert all(set(row) <= ALLOWED_CANDIDATE_KEYS for row in data["candidates"])
    assert all(set(row) <= ALLOWED_FILE_KEYS for row in data["files"])
    assert reviewer in page
    return data


def readme(reviewer, port, full_hours):
    label = reviewer[-1]
    return f"""# Natural-P3 phase-1 blind review — reviewer {label}

This kit is locked to `{reviewer}`. Do not open the other reviewer's kit or
responses. Do not inspect tracker paths, errors, event labels, P3 reports, or
project result files while reviewing.

1. Run `start_review.ps1`, or open `review.html` directly.
2. Review all 41 candidate boundaries as accept/reject/uncertain and write a
   concrete reason for every decision.
3. Review all 29 complete performances for a missed restart/take boundary and
   answer yes/no/uncertain with a reason.
4. Export `candidates.csv` and `files.csv`. Keep their row order and names.
5. Return only those two completed CSV files for placement in
   `results/p3_natural_boundary_review_packet/{reviewer}/`.

The candidate excerpts total 16.4 minutes. Complete-performance material totals
{full_hours:.2f} hours. Provisional units are deliberately absent from phase 1.
The local launcher binds only to `127.0.0.1:{port}` and does not upload data.
"""


def build_zip(folder, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(folder).as_posix())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_page = (PACKET / "review.html").read_text(encoding="utf8")
    base_data = embedded_data(base_page)
    full_hours = sum(float(row["duration_sec"]) for row in base_data["files"]) / 3600
    result = {
        "scope": "Reviewer-isolated tracker-blind phase-1 distribution kits",
        "source_html_sha256": digest(PACKET / "review.html"),
        "candidate_count": 41,
        "file_count": 29,
        "candidate_clip_minutes": 16.4,
        "full_performance_hours_per_reviewer": full_hours,
        "tracker_outcomes_included": False,
        "provisional_units_included": False,
        "kits": {},
    }
    for offset, reviewer in enumerate(REVIEWERS):
        folder = OUT / reviewer
        folder.mkdir(parents=True, exist_ok=True)
        page = customize_html(base_page, reviewer)
        validate_payload(page, reviewer)
        (folder / "review.html").write_text(page, encoding="utf8")
        copy_template(PACKET / "candidates.csv", folder / "candidates.csv", reviewer)
        copy_template(PACKET / "files.csv", folder / "files.csv", reviewer)
        shutil.copytree(PACKET / "media", folder / "media", dirs_exist_ok=True)
        port = 8765 + offset
        (folder / "README.md").write_text(readme(reviewer, port, full_hours), encoding="utf8")
        launcher = (
            "$kit = Split-Path -Parent $MyInvocation.MyCommand.Path\n"
            f"Start-Process 'http://127.0.0.1:{port}/review.html'\n"
            f"python -m http.server {port} --bind 127.0.0.1 --directory $kit\n"
        )
        (folder / "start_review.ps1").write_text(launcher, encoding="utf8")
        archive = OUT / f"{reviewer}_phase1.zip"
        build_zip(folder, archive)
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        result["kits"][reviewer] = {
            "directory": folder.relative_to(ROOT).as_posix(),
            "archive": archive.relative_to(ROOT).as_posix(),
            "archive_sha256": digest(archive),
            "html_sha256": digest(folder / "review.html"),
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
            "local_storage_key": f"p3-natural-review-{reviewer[-1]}",
        }
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
