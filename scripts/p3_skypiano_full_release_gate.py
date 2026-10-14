#!/usr/bin/env python3
"""Metadata-only gate for the public SKY-Piano full release."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_skypiano_full_release_gate"
PROJECT = "https://joonhyungbae.github.io/skypiano/"
EXPLORER_MANIFEST = "https://joonhyungbae.github.io/skypiano/explorer/Data/manifest.json"
GITHUB_REPOS = "https://api.github.com/users/joonhyungbae/repos?per_page=100"
KNOWN_SAMPLE_COUNT = 5
ARCHIVE_MARKERS = ("zenodo.org", "huggingface.co", "figshare.com", "osf.io", "dataverse", ".zip", ".tar")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "p3-release-gate/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def extract_urls(text):
    return sorted(set(re.findall(r"https?://[^\s\"'<>\\]+", text)))


def assess(project_html, explorer, repos):
    samples = explorer.get("samples", [])
    archive_urls = [url for url in extract_urls(project_html) if any(marker in url.lower() for marker in ARCHIVE_MARKERS)]
    likely_repos = [repo["html_url"] for repo in repos if "sky" in repo.get("name", "").lower() and "piano" in repo.get("name", "").lower()]
    enumerated_full_trials = len(samples) > KNOWN_SAMPLE_COUNT
    resolvable_release_pointer = bool(archive_urls or likely_repos or enumerated_full_trials)
    return {
        "explorer_sample_count": len(samples),
        "explorer_dataset_entries": len(explorer.get("datasets", [])),
        "archive_urls": archive_urls,
        "public_skypiano_repositories": likely_repos,
        "enumerated_full_trials": enumerated_full_trials,
        "resolvable_release_pointer": resolvable_release_pointer,
        "full_release_ready": bool(resolvable_release_pointer and enumerated_full_trials),
    }


def main():
    project_bytes = fetch(PROJECT)
    explorer_bytes = fetch(EXPLORER_MANIFEST)
    repos_bytes = fetch(GITHUB_REPOS)
    explorer = json.loads(explorer_bytes)
    repos = json.loads(repos_bytes)
    gate = assess(project_bytes.decode("utf8", "ignore"), explorer, repos)
    result = {
        "scope": "metadata-only SKY-Piano release gate; no data or tracker outcomes",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "project": {"url": PROJECT, "sha256": sha256(project_bytes)},
            "explorer_manifest": {"url": EXPLORER_MANIFEST, "sha256": sha256(explorer_bytes)},
            "github_repositories": {"url": GITHUB_REPOS, "sha256": sha256(repos_bytes)},
        },
        "gate": gate,
        "next_action": "freeze two-axis metadata split" if gate["full_release_ready"] else "wait for resolvable full trial manifest/archive",
        "temporal_outcomes_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
