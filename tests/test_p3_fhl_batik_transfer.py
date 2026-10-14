import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhl_batik_transfer.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_frozen_transfer_parameters():
 assert (m.WINDOW_SEC,m.HORIZON_SEC,m.CAP)==(12.,.5,5.)
 assert m.RADIUS==.5211881394162992 and m.FAMILIES==('K.279','K.331','K.332')
