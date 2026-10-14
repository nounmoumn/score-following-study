import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_sky_piano_sample_preflight.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_preflight_files_and_overlap_are_fixed():
 assert m.FILES==('midi.mid','score.mxl','alignment.json')
 assert 'Schubert_Op90_No4' in m.KNOWN_OVERLAP
