import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "p3_r_repeat_structure_audit.py"
SPEC = importlib.util.spec_from_file_location("repeat_structure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_detects_explicit_repeat_ending_and_jump():
    xml = b'''<score-partwise><part><measure number="1"><barline><repeat direction="forward"/></barline></measure>
    <measure number="2"><direction><direction-type><words>D.S. al Coda</words><segno/></direction-type>
    <sound dalsegno="segno1" tocoda="coda1"/></direction><barline><ending number="1" type="start"/>
    <repeat direction="backward" times="2"/></barline></measure></part></score-partwise>'''
    result = MODULE.audit_xml(xml)
    assert result["has_repeat_pair"] and result["has_ending"] and result["has_jump"]
    assert result["structure_eligible"]
    assert result["counts"]["repeat_backward_with_times"] == 1


def test_measure_numbers_alone_are_not_structure():
    result = MODULE.audit_xml(b'<score-partwise><part><measure number="1"/></part></score-partwise>')
    assert result["counts"]["measures"] == 1
    assert not result["structure_eligible"]

