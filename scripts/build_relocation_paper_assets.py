"""Build paper figures and machine-readable tables from frozen result JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    dev = read("results/online_relocation_validation/result.json")
    lock = read("results/online_relocation_validation_lockbox/result.json")
    audio = read("results/online_relocation_vienna_audio/result.json")
    rescue = read("results/online_relocation_audio_mean_evidence/result.json")

    # Table 1: the exact paper-level evidence boundary.
    rows = [
        ["MIDI development", dev["eligible_events"], dev["works"], 2, "FAIL", dev["work_equal"]["global_minus_delayed"], *dev["work_equal"]["global_minus_delayed_ci"]],
        ["MIDI lockbox", lock["eligible_events"], lock["works"], 2, "PASS", lock["work_equal"]["global_minus_delayed"], *lock["work_equal"]["global_minus_delayed_ci"]],
        ["Audio frozen pilot", audio["events"], audio["works"], len(audio["methods"]), "FAIL", audio["work_equal_improvement"], *audio["bootstrap95"]],
        ["Audio prospective rescue", rescue["events"], rescue["works"], len(rescue["methods"]), "FAIL", rescue["work_equal_improvement"], *rescue["bootstrap95"]],
    ]
    with (TABLES / "relocation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "events", "works", "trackers_with_events", "gate", "regret_improvement", "ci_low", "ci_high"])
        writer.writerows(rows)

    # Figure 1: lockbox per-work regret (the confirmatory positive result).
    work_rows = lock["per_work"]
    short_names = {
        "Beethoven/Piano_Sonatas/30": "Beethoven Sonata 30",
        "Beethoven/Piano_Sonatas/31": "Beethoven Sonata 31",
        "Beethoven/Piano_Sonatas/5": "Beethoven Sonata 5",
        "Chopin/Etudes_op_10/7": "Chopin Etude op.10/7",
        "Chopin/Etudes_op_25/10": "Chopin Etude op.25/10",
        "Chopin/Polonaises/53": "Chopin Polonaise 53",
        "Chopin/Sonata_3": "Chopin Sonata 3",
    }
    labels = [short_names.get(row["work"], row["work"]) for row in work_rows]
    y = np.arange(len(work_rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter([r["global_regret"] for r in work_rows], y, color="#D65F5F", s=46, label="Immediate accept")
    ax.scatter([r["delayed_regret"] for r in work_rows], y, color="#2878B5", s=46, label="2 s validator")
    for idx, row in enumerate(work_rows):
        ax.plot([row["delayed_regret"], row["global_regret"]], [idx, idx], color="#A0A0A0", lw=1.3, zorder=0)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decision regret (score-seconds; lower is better)")
    ax.set_title("Unseen MIDI works: delayed validation never increased regret")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "figure1_midi_lockbox_regret")

    # Figure 2: prospective audio result, where the parameter-free rescue worsened regret.
    audio_rows = rescue["per_work"]
    x = np.arange(len(audio_rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x - width / 2, [r["global_regret"] for r in audio_rows], width, color="#7A7A7A", label="Accept relocation")
    ax.bar(x + width / 2, [r["mean_evidence_regret"] for r in audio_rows], width, color="#D65F5F", label="Audio mean-evidence")
    ax.set_xticks(x, [r["work"].replace("/", "\n") for r in audio_rows])
    ax.set_ylabel("Mean decision regret (score-seconds)")
    ax.set_title("Prospective audio rescue: no gain and one clear regression")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "figure2_audio_rescue_regret")

    # Figure 3: event-support funnel, emphasizing tracker and work concentration.
    labels = ["MIDI dev\n(7 works)", "MIDI lockbox\n(7 works)", "Audio pilot\n(2 works)", "Audio prospective\n(2 works)"]
    events = [dev["eligible_events"], lock["eligible_events"], audio["events"], rescue["events"]]
    colors = ["#9ABBD7", "#2878B5", "#E6A57E", "#D65F5F"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(np.arange(4), events, color=colors)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Eligible relocation events")
    ax.set_title("Audio transfer changed the support, not only the accuracy")
    ax.bar_label(bars, padding=3)
    ax.text(2.5, 20, "audio: outerhmm only\n0 Arzt events", ha="center", va="center", fontsize=10, color="#8B2E2E")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "figure3_event_support")

    manifest = {
        "source_results": [
            "results/online_relocation_validation/result.json",
            "results/online_relocation_validation_lockbox/result.json",
            "results/online_relocation_vienna_audio/result.json",
            "results/online_relocation_audio_mean_evidence/result.json",
        ],
        "figures": sorted(p.name for p in FIGURES.iterdir()),
        "tables": sorted(p.name for p in TABLES.iterdir()),
    }
    (PAPER / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
