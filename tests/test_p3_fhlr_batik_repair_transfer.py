import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhlr_batik_repair_transfer.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_repair_limit_is_frozen():assert m.MAX_CORRECTION==.15
