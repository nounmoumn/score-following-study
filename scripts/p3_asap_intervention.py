"""Paired position-evidence ablation on actual ASAP MIDI features."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.ground_truth import piecewise_linear_gt
from score_following.evidence_ablation import replay_error
from score_following.prospective import bootstrap_mean


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/p3_asap_intervention.yaml")
    args = p.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out = ROOT / "results" / f"{cfg['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    with open(ROOT / cfg["manifest"], encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    root = ROOT / cfg["asap_root"]
    annot = json.loads((root / "asap_annotations.json").read_text(encoding="utf-8"))
    with open(root / "metadata.csv", encoding="utf-8") as f:
        meta = {r["midi_performance"]: r for r in csv.DictReader(f)}
    files = [cfg_path, Path(__file__), ROOT / cfg["manifest"],
             ROOT / "P3_REVIVAL_PROTOCOL.md", root / "asap_annotations.json",
             ROOT / "src/score_following/evidence_ablation.py",
             ROOT / "src/score_following/online_dtw.py", ROOT / "src/score_following/prospective.py",
             ROOT / "src/score_following/features.py"]
    for m in manifest:
        files += [root / m["performance"], root / meta[m["performance"]]["midi_score"]]
    provenance = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out / "provenance.json").write_text(json.dumps(dict(python=sys.version, numpy=np.__version__,
        platform=platform.platform(), sha256=provenance), indent=2), encoding="utf-8")
    trials, traces = [], []
    hop = cfg["hop_sec"]
    onset, restore = [int(round(cfg[k] / hop)) for k in ("onset_sec", "restore_sec")]
    log_file = open(out / "run.log", "w", encoding="utf-8")
    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()
    log(f"output: {out}")
    for index, m in enumerate(manifest):
        r = meta[m["performance"]]
        pn, sn = load_notes(root / r["midi_performance"]), load_notes(root / r["midi_score"])
        pd, sd = max(n[1] for n in pn), max(n[1] for n in sn)
        hop_v = hop * sd / pd
        if abs(hop_v - float(m["hop_score_sec"])) > 1e-10:
            raise ValueError("manifest tempo scale does not match source MIDI")
        if pd < cfg["duration_sec"]:
            raise ValueError("preassigned excerpt not available; do not silently exclude")
        u = notes_to_chroma_stream(pn, cfg["duration_sec"], hop)
        v = notes_to_chroma_stream(sn, sd, hop_v)
        a = annot[r["midi_performance"]]
        gt, lo, hi = piecewise_linear_gt(a["performance_beats"], a["midi_score_beats"])
        times = np.arange(len(u)) * hop
        truth = gt(times) / hop_v
        # Use score seconds, as in the original P1/P2 errors.
        intact = replay_error(u, v, truth, hop=hop_v, band=int(round(cfg["band_sec"] / hop)))
        erased = replay_error(u, v, truth, hop=hop_v, band=int(round(cfg["band_sec"] / hop)), erase=(onset, restore))
        valid = (times >= lo) & (times <= hi)
        pre = valid & (times < cfg["onset_sec"])
        during = valid & (times >= cfg["onset_sec"]) & (times < cfg["restore_sec"])
        after = valid & (times >= cfg["restore_sec"]) & (times < cfg["recovery_end_sec"])
        if during.sum() != restore-onset or after.sum() != int(round((cfg["recovery_end_sec"]-cfg["restore_sec"])/hop)):
            raise ValueError("incomplete annotated intervention interval")
        diff = erased-intact
        trial = dict(composition=m["composition"], performance=m["performance"],
            pre_max_difference_sec=float(np.max(abs(diff[pre]))),
            intact_error_sec=float(intact[during].mean()), erased_error_sec=float(erased[during].mean()),
            during_effect_sec=float(diff[during].mean()), after_effect_sec=float(diff[after].mean()))
        trials.append(trial)
        for i in np.flatnonzero(valid):
            traces.append(dict(composition=m["composition"], performance=m["performance"], time_sec=float(times[i]),
                               intact_error_sec=float(intact[i]), erased_error_sec=float(erased[i])))
        log(f"{index+1}/{len(manifest)} {m['performance']}: during_effect={trial['during_effect_sec']:+.6f}s")
        write_csv(out / "performance_results.csv", trials)
    compositions = []
    for name in sorted(set(t["composition"] for t in trials)):
        subset = [t for t in trials if t["composition"] == name]
        row = dict(composition=name, performances=len(subset))
        row.update({key: float(np.mean([t[key] for t in subset])) for key in
                    ("intact_error_sec", "erased_error_sec", "during_effect_sec", "after_effect_sec")})
        compositions.append(row)
    effect = np.array([r["during_effect_sec"] for r in compositions])
    after_effect = np.array([r["after_effect_sec"] for r in compositions])
    metrics = dict(performances=len(trials), compositions=len(compositions),
        during_effect_sec=float(effect.mean()), ci95=bootstrap_mean(effect, cfg["seed"], cfg["n_bootstrap"]),
        positive_compositions=int((effect>0).sum()),
        pre_max_difference_sec=max(r["pre_max_difference_sec"] for r in trials),
        after_effect_sec=float(after_effect.mean()), after_ci95=bootstrap_mean(after_effect, cfg["seed"], cfg["n_bootstrap"]),
        intact_error_sec=float(np.mean([r["intact_error_sec"] for r in compositions])),
        erased_error_sec=float(np.mean([r["erased_error_sec"] for r in compositions])),
        scope="Artificial position-cost erasure on real ASAP MIDI; not natural ambiguity mediation.")
    metrics["pass"] = metrics["ci95"][0] > 0 and metrics["pre_max_difference_sec"] == 0
    write_csv(out / "composition_results.csv", compositions)
    write_csv(out / "traces.csv", traces)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps(metrics, indent=2))
    log_file.close()


if __name__ == "__main__":
    main()
