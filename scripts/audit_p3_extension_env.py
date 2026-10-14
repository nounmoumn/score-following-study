"""Read-only availability check, no experimental outcomes."""
import csv
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ["numpy", "scipy", "numba", "pytest", "matplotlib"]:
    print(name, bool(importlib.util.find_spec(name)))
with open(root / "results/p3_revival_20260917_230027/manifest.csv", encoding="utf-8") as f:
    old = list(csv.DictReader(f))
seen = {r["composition"] for r in old}
with open(root / "asap_dataset/metadata.csv", encoding="utf-8") as f:
    meta = list(csv.DictReader(f))
annot = json.loads((root / "asap_dataset/asap_annotations.json").read_text(encoding="utf-8"))
eligible = set()
for r in meta:
    label = r['composer'] + '/' + r['title']
    a = annot.get(r["midi_performance"], {})
    b = a.get("performance_beats", [])
    if label not in seen and a.get("score_and_performance_aligned") and b and min(b) < 10 and max(b)>91:
        eligible.add(label)
print("old performances/compositions", len(old), len(seen))
print("new compositions with annotations beyond 91s", len(eligible))
for item in old:
    beats = annot[item["performance"]]["performance_beats"]
    if min(beats)>10 or max(beats)<90:
        print("old coverage exception", item["performance"], min(beats), max(beats))
