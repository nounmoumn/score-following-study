import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_m_pianocore_transfer.py"
SPEC = importlib.util.spec_from_file_location("p3_m_transfer", SCRIPT)
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


def rows():
    output = []
    for composer in range(25):
        for landmark in range(12):
            ambiguity = landmark / 11 + composer * .001
            output.append({"composer": f"c{composer:02d}", "ambiguity": ambiguity,
                           "weakness_local": ambiguity + .01 * np.sin(landmark),
                           "time_sec": 20 + 5 * landmark})
    return output


def test_composer_effects_recover_relationship():
    effects = transfer.composer_effects(rows())
    assert len(effects) == 25
    assert min(r["rho"] for r in effects) > .9


def test_shift_preserves_composer_values():
    source = rows()
    shifted = transfer.shifted_ambiguity(source, np.random.default_rng(2))
    for composer in sorted({r["composer"] for r in source}):
        indices = [i for i, r in enumerate(source) if r["composer"] == composer]
        assert np.allclose(sorted(shifted[indices]), sorted(source[i]["ambiguity"] for i in indices))

