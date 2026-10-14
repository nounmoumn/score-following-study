"""Run the complete GP -> annotation -> MIDI -> REAPER -> WAV -> QA pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n>>> " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder containing .gp/.gp3/.gp4/.gp5")
    parser.add_argument("--dataset", type=Path, default=Path("data/guitar_dataset"))
    parser.add_argument(
        "--template", type=Path, default=Path("templates/shreddage_stratus_template.rpp")
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--max-fret", type=int, default=12)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    python = sys.executable

    run(
        [
            python,
            str(scripts / "build_guitar_dataset.py"),
            str(args.input),
            "--output",
            str(args.dataset),
            "--max-fret",
            str(args.max_fret),
        ]
    )
    prepare = [
        python,
        str(scripts / "prepare_reaper_batch.py"),
        str(args.template),
        "--dataset",
        str(args.dataset),
    ]
    if args.include_partial:
        prepare.append("--include-partial")
    for item_id in args.only_id:
        prepare.extend(["--only-id", item_id])
    if args.render:
        prepare.append("--render")
    run(prepare)
    if args.render:
        run(
            [
                python,
                str(scripts / "validate_guitar_renders.py"),
                "--dataset",
                str(args.dataset),
            ]
        )


if __name__ == "__main__":
    main()
