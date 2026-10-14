import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_e_formula_audit.py"
SPEC = importlib.util.spec_from_file_location("p3_e_formula_audit", SCRIPT)
formula = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formula)


def test_transition_matrix_is_column_stochastic_with_end_absorption():
    matrix = formula.transition_matrix(19)
    assert np.allclose(matrix.sum(axis=0), 1)
    assert np.isclose(matrix[-1, -1], 1)


def test_erasure_recurrence_and_closed_form_are_exact():
    result = formula.audit(seed=7, n=29, frames=17)
    assert result["all_exact_within_1e_12"] is True

