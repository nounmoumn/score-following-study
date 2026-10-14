"""Classify nonlocal switches as observation-led, shock-led, or dynamics-led."""

import hashlib
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import partitura as pt
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from score_following.temporal import healthy_before
from p3_batik_external_run import load_gt
from p3_short_signal import sustained_start

OUT = ROOT / "results/p3_switch_recovery"
CONFIG = ROOT / "configs/p3_temporal.yaml"
PROTOCOL = ROOT / "P3_SWITCH_RECOVERY_PROTOCOL.md"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    batik = json.loads((ROOT / "results/p3_batik_external/manifest.json").read_text(encoding="utf-8"))
    kraisler = json.loads((ROOT / "results/p3_kraisler_signal/manifest.json").read_text(encoding="utf-8"))
    result = []
    for row in batik["rows"]:
        if row["eligible"]: result.append(dict(row, dataset="Batik", case_id=f"Batik/{row['sonata']}/{row['movement']}", max_duration=90.))
    for row in kraisler["rows"]:
        if row["eligible"]: result.append(dict(row, dataset="KRAISLER", case_id=f"KRAISLER/{row['track']}", max_duration=None))
    return result


def load_case(case, hop):
    performance = ROOT / case["performance"]; score_path = ROOT / case["score"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(str(score_path))
        notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_duration = float((notes["onset_beat"] + notes["duration_beat"]).max()) * .5
    pnotes = load_notes(performance); full_duration = max(n[1] for n in pnotes)
    duration = min(full_duration, case["max_duration"]) if case["max_duration"] else full_duration
    score_hop = hop * score_duration / full_duration
    snotes = [(float(r["onset_beat"]) * .5, float(r["onset_beat"] + r["duration_beat"]) * .5, int(r["pitch"])) for r in notes]
    u = notes_to_chroma_stream(pnotes, duration, hop)
    v = notes_to_chroma_stream(snotes, score_duration, score_hop)
    beats, perf_times = load_gt(ROOT / case["match"])
    time = np.arange(len(u)) * hop
    truth = np.interp(time, perf_times, beats * .5) / score_hop
    return u, v, truth, score_hop


def region_mean(likelihood, center, radius):
    center = min(len(likelihood) - 1, max(0, int(round(center))))
    lo = max(0, center - radius); hi = min(len(likelihood), center + radius + 1)
    return float(likelihood[lo:hi].mean())


def evidence(cost, i, old_end, new_end, radius, beta, expected_step):
    values = []
    for j in range(i - 20, i):
        back = i - j
        old_center = old_end - back * expected_step
        new_center = new_end - back * expected_step
        likelihood = np.exp(-beta * (cost[j] - cost[j].min()))
        if np.all(likelihood == likelihood[0]): value = 0.
        else: value = float(np.log(region_mean(likelihood, new_center, radius)) - np.log(region_mean(likelihood, old_center, radius)))
        values.append(value)
    likelihood = np.exp(-beta * (cost[i] - cost[i].min()))
    if np.all(likelihood == likelihood[0]): switch_lr = 0.
    else: switch_lr = float(np.log(region_mean(likelihood, new_end, radius)) - np.log(region_mean(likelihood, old_end, radius)))
    return float(sum(values)), switch_lr


def recovered(error, i):
    healthy = error[i:min(len(error), i + 21)] <= .5
    return any(bool(np.all(healthy[j:j + 5])) for j in range(max(0, len(healthy) - 4)))


def rate(rows, kind):
    selected = [r for r in rows if r["kind"] == kind]
    return sum(r["persistent"] for r in selected) / len(selected) if selected else None


def bootstrap_difference(rows, seed=20260920, n=10000):
    grouped = defaultdict(list)
    for row in rows: grouped[row["case_id"]].append(row)
    keys = sorted(grouped); rng = np.random.default_rng(seed); values = []
    for _ in range(n):
        sample = [r for key in rng.choice(keys, len(keys), replace=True) for r in grouped[key]]
        a = [r["persistent"] for r in sample if r["kind"] == "observation_led"]
        b = [r["persistent"] for r in sample if r["kind"] != "observation_led"]
        if a and b: values.append(float(np.mean(a) - np.mean(b)))
    return [float(x) for x in np.quantile(values, [.025, .5, .975])] if values else [None] * 3


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); hop = float(cfg["hop_sec"])
    bayes = cfg["bayes"]; transition = np.asarray(bayes["transition"], dtype=float)
    expected_step = float(np.arange(len(transition)) @ transition); local_max = len(transition) - 1
    rows = []; case_counts = []
    all_cases = cases()
    with threadpool_limits(limits=1):
        for number, case in enumerate(all_cases, 1):
            u, v, truth, score_hop = load_case(case, hop); cost = cosine_cost(u, v)
            path = bayes_from_cost(cost, transition, bayes["emission_beta"], bayes["restart_probability"])
            error = np.abs(path - truth) * score_hop
            radius = max(1, int(round(.5 / score_hop))); threshold = max(local_max, radius)
            raw = np.flatnonzero(np.abs(np.diff(path, prepend=path[0])) > threshold)
            selected = []; last = -10_000
            for i in raw:
                if i < 20 or i + 20 >= len(path) or i - last < 20: continue
                if not healthy_before(error, i - 1, 20, .5): continue
                selected.append(int(i)); last = int(i)
            for i in selected:
                old_end = float(path[i - 1] + expected_step); new_end = float(path[i])
                c_pre, switch_lr = evidence(cost, i, old_end, new_end, radius, bayes["emission_beta"], expected_step)
                if c_pre > 0: kind = "observation_led"
                elif switch_lr > 0: kind = "shock_led"
                else: kind = "dynamics_led"
                onset = sustained_start(error, i, i + 20)
                has_recovered = recovered(error, i)
                rows.append({"dataset": case["dataset"], "case_id": case["case_id"], "time_sec": i * hop,
                             "jump_states": int(path[i] - path[i - 1]), "jump_score_sec": float((path[i] - path[i - 1]) * score_hop),
                             "C_pre": c_pre, "switch_LR": switch_lr, "kind": kind,
                             "new_failure": onset is not None, "onset_sec": onset * hop if onset is not None else None,
                             "recovered": has_recovered, "persistent": not has_recovered})
            case_counts.append({"case_id": case["case_id"], "dataset": case["dataset"], "switches": len(selected)})
            print(f"[{number}/{len(all_cases)}] {case['case_id']}: {len(selected)} switches")
    counts = Counter(r["kind"] for r in rows); persistent = [r for r in rows if r["persistent"]]
    persistent_counts = Counter(r["kind"] for r in persistent)
    rates = {kind: rate(rows, kind) for kind in ("observation_led", "shock_led", "dynamics_led")}
    ci = bootstrap_difference(rows)
    enough = len(rows) >= 20 and len(persistent) >= 5
    obs_share = persistent_counts["observation_led"] / len(persistent) if persistent else None
    dyn_share = persistent_counts["dynamics_led"] / len(persistent) if persistent else None
    others = [r["persistent"] for r in rows if r["kind"] != "observation_led"]
    obs = [r["persistent"] for r in rows if r["kind"] == "observation_led"]
    obs_difference = float(np.mean(obs) - np.mean(others)) if obs and others else None
    p3 = enough and obs_difference >= .2 and ci[0] is not None and ci[0] > 0 and obs_share >= .5
    dyn_difference = (rates["dynamics_led"] - rates["observation_led"]
                      if rates["dynamics_led"] is not None and rates["observation_led"] is not None else None)
    dynamics = enough and dyn_difference is not None and dyn_difference >= .2 and dyn_share >= .5
    decision = "advance_narrow_p3" if p3 else "dynamics_lock_in_supported" if dynamics else "mixed_or_below_gate" if enough else "insufficient_information"
    result = {"scope": "Batik and KRAISLER development; no independent confirmation",
              "cases": len(all_cases), "switches": len(rows), "persistent_recovery_failures": len(persistent),
              "new_failure_switches": sum(r["new_failure"] for r in rows),
              "kind_counts": dict(counts), "persistent_kind_counts": dict(persistent_counts),
              "persistent_rates": rates, "observation_led_minus_others": obs_difference,
              "observation_difference_cluster_bootstrap_95": ci,
              "persistent_observation_share": obs_share, "persistent_dynamics_share": dyn_share,
              "dynamics_minus_observation_persistent_rate": dyn_difference,
              "decision": decision, "advance_to_smd": bool(p3),
              "case_counts": case_counts, "rows": rows,
              "provenance": {"protocol_sha256": sha256(PROTOCOL), "config_sha256": sha256(CONFIG),
                             "script_sha256": sha256(Path(__file__))}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "case_counts", "provenance")}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
