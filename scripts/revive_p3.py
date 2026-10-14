"""Frozen P3 rescue protocol: python scripts/revive_p3.py --config configs/p3_revival.yaml."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes
from score_following.ground_truth import piecewise_linear_gt
from score_following.prospective import (completed_window_values, forecast_comparison,
                                        bootstrap_mean)
from score_following.evidence_ablation import synthetic_pair, replay_error


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def diverse(rows):
    by_composer = defaultdict(list)
    for r in rows:
        by_composer[r["composer"]].append(r)
    return [lst[i] for i in range(max(map(len, by_composer.values())))
            for lst in by_composer.values() if i < len(lst)]


def prepare_landmarks(cfg, log):
    source = ROOT / cfg["source"]
    original_cfg = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    if original_cfg["ambiguity_metric"] != "maxsim":
        raise ValueError("only unmodified maxsim cache supported")
    hop = original_cfg["feature"]["frame_hop_sec"]
    root = ROOT / cfg["asap_root"]
    annot = json.loads((root / "asap_annotations.json").read_text(encoding="utf-8"))
    meta = read_csv(root / "metadata.csv")
    if original_cfg["data"].get("diverse_sampling", True):
        meta = diverse(meta)
    meta = [r for r in meta if annot.get(r["midi_performance"], {}).get(
        "score_and_performance_aligned", False) and (root / r["midi_performance"]).exists()
        and (root / r["midi_score"]).exists()]
    diag = read_csv(source / "piece_diagnostics.csv")
    frames = read_csv(source / "frame_results.csv")
    if sum(int(d["n_frames"]) for d in diag) != len(frames):
        raise ValueError("cache diagnostics/frame count mismatch")
    lm = cfg["landmark"]
    rows, manifest, cursor = [], [], 0
    for index, d in enumerate(diag):
        # Fail closed if earlier source execution skipped an unexpected item.
        r = meta[index]
        label = f"{r['composer']}/{r['title']}"
        if label != d["label"]:
            raise ValueError("metadata order differs from cache; explicit manifest required")
        n = int(d["n_frames"])
        segment = frames[cursor:cursor+n]
        cursor += n
        if any(s["label"] != label for s in segment):
            raise ValueError("cache boundary/label mismatch")
        time = np.array([float(s["perf_time_sec"]) for s in segment])
        if np.any(np.diff(time) <= 0):
            raise ValueError("merged performances or nonmonotone timestamps")
        err = np.array([float(s["online_error_sec"]) for s in segment])
        amb = np.array([float(s["ambiguity"]) for s in segment])
        a = annot[r["midi_performance"]]
        gt, _, _ = piecewise_linear_gt(a["performance_beats"], a["midi_score_beats"])
        perf_dur = max(note[1] for note in load_notes(root / r["midi_performance"]))
        score_dur = max(note[1] for note in load_notes(root / r["midi_score"]))
        hop_v = hop * score_dur / perf_dur
        window = max(2, int(round(original_cfg["feature"]["ambiguity_window_sec"] / hop_v)))
        score_t = gt(time)
        ends = (np.rint(score_t / hop_v).astype(int) + window) * hop_v
        causal = completed_window_values(score_t, ends, amb)
        first = np.ceil((time[0] + lm["history_sec"] + 3) / lm["stride_sec"]) * lm["stride_sec"]
        count = 0
        for t in np.arange(first, time[-1] - max(lm["horizons_sec"]), lm["stride_sec"]):
            queries = [t, t - 2, t - 5] + [t + h for h in lm["horizons_sec"]]
            idx = np.searchsorted(time, queries)
            if np.any(idx >= len(time)) or np.max(abs(time[idx] - queries)) > hop * 0.1:
                continue
            i, p2, p5 = idx[:3]
            # Do not bridge missing stretches, even if endpoint samples exist.
            if np.max(np.diff(time[p5:idx.max()+1])) > hop * 1.1:
                continue
            if not np.isfinite(causal[[i, p2]]).all():
                continue
            row = dict(composition=label, performance=r["midi_performance"], time_sec=float(t),
                       log_error=float(np.log1p(err[i])),
                       log_error_past2=float(np.log1p(err[p2])),
                       log_error_past5=float(np.log1p(err[p5])), elapsed_min=float(t / 60),
                       past_tempo=float((score_t[i] - score_t[p2]) / 2),
                       ambiguity_completed=float(causal[i]),
                       ambiguity_change2=float(causal[i] - causal[p2]))
            for h, k in zip(lm["horizons_sec"], idx[3:]):
                row[f"target_{h:g}s"] = float(np.log1p(err[k]) - np.log1p(err[i]))
            rows.append(row)
            count += 1
        manifest.append(dict(composition=label, performance=r["midi_performance"],
                             cached_frames=n, landmarks=count, hop_score_sec=hop_v,
                             window_frames=window))
        log(f"cache {index+1}/{len(diag)}: {label}, landmarks={count}")
    if any(m["landmarks"] < 4 for m in manifest):
        raise ValueError("insufficient landmarks; do not silently filter a performance")
    return rows, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p3_revival.yaml")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = ROOT / "results" / f"{cfg['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    log_file = open(out / "run.log", "w", encoding="utf-8")
    def log(s):
        print(s, flush=True)
        log_file.write(s + "\n")
        log_file.flush()
    log(f"output: {out}")
    files = [cfg_path, Path(__file__), ROOT / "P3_REVIVAL_PROTOCOL.md",
             ROOT / "src/score_following/prospective.py",
             ROOT / "src/score_following/evidence_ablation.py",
             ROOT / "src/score_following/online_dtw.py",
             ROOT / cfg["source"] / "frame_results.csv",
             ROOT / cfg["source"] / "piece_diagnostics.csv",
             ROOT / cfg["source"] / "config.yaml",
             ROOT / cfg["asap_root"] / "metadata.csv",
             ROOT / cfg["asap_root"] / "asap_annotations.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "provenance.json").write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
                                                       numpy=np.__version__, sha256=hashes), indent=2), encoding="utf-8")
    rows, manifest = prepare_landmarks(cfg, log)
    write_csv(out / "landmarks.csv", rows)
    write_csv(out / "manifest.csv", manifest)
    groups = np.array([r["composition"] for r in rows])
    perfs = np.array([r["performance"] for r in rows])
    base = np.array([[r[k] for k in ("log_error", "log_error_past2", "log_error_past5",
                                   "elapsed_min", "past_tempo")] for r in rows])
    extra = np.array([[r[k] for k in ("ambiguity_completed", "ambiguity_change2")] for r in rows])
    metrics, group_rows = {"natural": {}, "scope": "exploratory; oracle cache; synthetic intervention"}, []
    lm = cfg["landmark"]
    for h in lm["horizons_sec"]:
        y = np.array([r[f"target_{h:g}s"] for r in rows])
        summary, bp, fp, effects = forecast_comparison(base, extra, y, groups, perfs,
            alpha=lm["ridge_alpha"], seed=cfg["seed"], n_bootstrap=lm["n_bootstrap"], n_shifts=lm["n_shifts"])
        metrics["natural"][f"{h:g}s"] = summary
        for name, effect in zip(np.unique(groups), effects):
            group_rows.append(dict(horizon_sec=h, composition=name, mae_improvement_log1p=float(effect)))
        for row, b, f in zip(rows, bp, fp):
            row[f"baseline_prediction_{h:g}s"] = float(b)
            row[f"full_prediction_{h:g}s"] = float(f)
        log(f"natural {h:g}s: effect={summary['mae_improvement_log1p']:.6f}, "
            f"CI={summary['bootstrap_ci95']}, shift_p={summary['shift_p_upper']:.4f}")
    primary = metrics["natural"][f"{lm['primary_horizon_sec']:g}s"]
    metrics["natural_pass"] = (primary["bootstrap_ci95"][0] > 0 and primary["shift_p_upper"] <= .05)
    write_csv(out / "predictions.csv", rows)
    write_csv(out / "composition_results.csv", group_rows)
    # Preserve completed natural analysis before running intervention.
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ic = cfg["intervention"]
    trials, trace_rows = [], []
    onset, restore = ic["onset_frame"], ic["restore_frame"]
    for seed in range(ic["seeds"]):
        for speed in ic["speed_changes"]:
            u, v, truth = synthetic_pair(seed + cfg["seed"], speed, frames=ic["frames"], onset=onset)
            intact = replay_error(u, v, truth, hop=ic["hop_sec"], band=ic["band_frames"])
            erased = replay_error(u, v, truth, hop=ic["hop_sec"], band=ic["band_frames"], erase=(onset, restore))
            difference = erased - intact
            trial = dict(seed=seed, speed_change=speed,
                pre_max_difference_sec=float(np.max(abs(difference[:onset]))),
                intact_during_error_sec=float(intact[onset:restore].mean()),
                erased_during_error_sec=float(erased[onset:restore].mean()),
                during_effect_sec=float(difference[onset:restore].mean()),
                after_effect_sec=float(difference[restore:restore+100].mean()))
            trials.append(trial)
            for i in range(len(truth)):
                trace_rows.append(dict(seed=seed, speed_change=speed, time_sec=i*ic["hop_sec"],
                                       intact_error_sec=float(intact[i]), erased_error_sec=float(erased[i])))
        log(f"intervention seed {seed+1}/{ic['seeds']}")
    paired = np.array([np.mean([t["during_effect_sec"] for t in trials
                               if t["seed"] == s and t["speed_change"] != 0]) for s in range(ic["seeds"])])
    intervention = dict(paired_effect_sec=float(paired.mean()),
        bootstrap_ci95=bootstrap_mean(paired, cfg["seed"], lm["n_bootstrap"]),
        pre_max_difference_sec=max(t["pre_max_difference_sec"] for t in trials),
        positive_seeds=int((paired > 0).sum()), seeds=ic["seeds"], by_speed={})
    for speed in ic["speed_changes"]:
        subset = [t for t in trials if t["speed_change"] == speed]
        intervention["by_speed"][str(speed)] = {key: float(np.mean([t[key] for t in subset]))
            for key in ("during_effect_sec", "after_effect_sec", "intact_during_error_sec", "erased_during_error_sec")}
    metrics["intervention"] = intervention
    metrics["intervention_pass"] = intervention["bootstrap_ci95"][0] > 0 and intervention["pre_max_difference_sec"] == 0
    write_csv(out / "intervention_trials.csv", trials)
    write_csv(out / "intervention_traces.csv", trace_rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"intervention: {json.dumps(intervention)}")
    log(f"natural_pass={metrics['natural_pass']}; intervention_pass={metrics['intervention_pass']}")
    log_file.close()


if __name__ == "__main__":
    main()
