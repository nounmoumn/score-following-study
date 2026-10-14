"""Freeze planning metadata only; never read tracker outcomes or run experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.onset import work_family


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import numpy
    import scipy
    import numba
    import yaml

    out = ROOT / "output/score_priority_plan"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    if state_path.exists():
        raise SystemExit("Planning snapshot already exists; do not overwrite execution state.")
    source = ROOT / "results/p3_extension_20260917_234336/manifest.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    annotation_path = ROOT / "asap_dataset/asap_annotations.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    family = lambda row: work_family(str(Path(row["score"]).parent).replace("\\", "/"))
    dev_families = {family(row) for row in original if row["cohort"] == "development"}
    rows = []
    for row in original:
        work = family(row)
        if row["cohort"] == "development":
            role, reason = "primary_opened", ""
        elif work in dev_families:
            role, reason = "excluded_family_overlap", "same work family as primary"
        else:
            role, reason = "replication_opened_nonoverlap", ""
        perf = ROOT / "asap_dataset" / row["performance"]
        score = ROOT / "asap_dataset" / row["score"]
        rows.append({
            "case_id": row["case_id"], "original_cohort": row["cohort"],
            "role": role, "performance": row["performance"], "score": row["score"],
            "composition": row["composition"], "work_family": work,
            "composer": row["score"].split("/")[0],
            "exposure": "previously_opened_p3_not_fresh_holdout",
            "family_exclusion_reason": reason,
            "files_exist": perf.exists() and score.exists(),
            "annotation_aligned": annotation.get(row["performance"], {}).get("score_and_performance_aligned", False),
            "performance_sha256": sha256(perf) if perf.exists() else "",
            "score_sha256": sha256(score) if score.exists() else "",
        })
    manifest = out / "cohort_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config_path = ROOT / "configs/score_priority_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["data"]["fresh_holdout"] is False
    assert config["selection"]["primary_fraction"] == 0.20
    counts = {}
    for role in sorted({row["role"] for row in rows}):
        subset = [row for row in rows if row["role"] == role]
        counts[role] = {
            "performances": len(subset),
            "score_files": len({row["score"] for row in subset}),
            "work_families": len({row["work_family"] for row in subset}),
            "composers": dict(Counter(row["composer"] for row in subset)),
        }
    fixed_paths = [source, annotation_path, config_path, ROOT / "SCORE_PRIORITY_EXECUTION_PLAN.md",
                   ROOT / "SCORE_PRIORITY_HANDOFF.md", Path(__file__),
                   ROOT / "src/score_following/onset.py", ROOT / "src/score_following/online_dtw.py",
                   ROOT / "src/score_following/p3_extension.py", ROOT / "src/score_following/features.py"]
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in fixed_paths}
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "metadata_only_planning_snapshot", "counts": counts,
        "files_exist_all": all(row["files_exist"] for row in rows),
        "annotation_aligned_all": all(row["annotation_aligned"] for row in rows),
        "final_eligibility_verified": False, "tracker_outcomes_read": False,
        "experiments_run": False, "manifest_sha256": sha256(manifest), "source_hashes": hashes,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": numpy.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
        "limitations": ["Work grouping needs title/version review before outcome evaluation.",
                        "Past holdout is already opened and is not a fresh validation set.",
                        "GT clock, score-only features, causal replay, and utility remain unverified."],
    }
    (out / "audit_snapshot.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {
        "state": "PLANNED", "experimental_execution_authorized_in_this_turn": False,
        "completed": ["protocol", "config", "handoff", "metadata_and_file_hash_snapshot"],
        "next_step": "When asked to execute: implement S0 inventory with score_priority_inventory.py.",
        "experiment_scripts_implemented": False, "positive_result": None,
        "research_execution_wall_seconds": 0,
        "config_sha256": sha256(config_path), "manifest_sha256": sha256(manifest),
        "snapshot_command": ".venv\\Scripts\\python.exe scripts/score_priority_plan_snapshot.py",
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": state["state"], "counts": counts,
                      "all_files_exist": audit["files_exist_all"],
                      "all_annotation_flags_true": audit["annotation_aligned_all"],
                      "experiments_run": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
