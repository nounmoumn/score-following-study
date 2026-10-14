import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"p3_td_transfer.py"
SPEC=importlib.util.spec_from_file_location("p3_td_transfer",SCRIPT)
transfer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(transfer)


def test_transfer_import_does_not_require_partitura():
    assert transfer.duration.DURATIONS==(2,5,10)

