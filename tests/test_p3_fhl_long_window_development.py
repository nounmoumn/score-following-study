import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhl_long_window_development.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_extension_grid_and_gate_are_frozen():
 assert m.WINDOWS==(6.,8.,12.) and m.HORIZONS==(.5,1.) and m.CAP==5.
