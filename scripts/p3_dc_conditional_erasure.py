#!/usr/bin/env python3
"""Exact decomposition of erasure risk into future-projected observation help."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p3_d_dose_response as dose


def prediction_matrix(states: int, weights, restart=dose.RESTART):
    matrix = dose.transition_matrix(states, weights)
    return (1.0 - restart) * matrix + restart / states


def apply_prediction(vector, weights, restart=dose.RESTART):
    predicted = np.zeros_like(vector)
    for step, weight in enumerate(weights):
        if step == 0:
            predicted += weight * vector
        else:
            predicted[step:] += weight * vector[:-step]
            predicted[-1] += weight * vector[-step:].sum()
    return (1.0 - restart) * predicted + restart * vector.sum() / len(vector)


def decompose(initial, costs, truth, score_hop, weights, cap=5.0):
    states = len(initial)
    intact = np.asarray(initial, dtype=float).copy()
    erased = intact.copy(); propagated = np.zeros(states)
    help_terms = []
    for row in costs:
        prior = apply_prediction(intact, weights)
        likelihood = np.exp(-dose.BETA * (row - row.min()))
        intact = prior * likelihood; intact /= intact.sum()
        erased = apply_prediction(erased, weights)
        delta = intact - prior
        propagated = apply_prediction(propagated, weights) + delta
    risk_vector = np.minimum(np.abs(np.arange(states) - truth[-1]) * score_hop, cap)
    posterior_residual = float(np.max(np.abs((intact - erased) - propagated)))
    erasure_effect = float(risk_vector @ (erased - intact))
    projected_help_sum = -float(risk_vector @ propagated)
    risk_residual = float(abs(erasure_effect - projected_help_sum))
    return {"posterior_residual": posterior_residual, "risk_residual": risk_residual,
            "erasure_effect": erasure_effect, "projected_help_sum": projected_help_sum}


def random_audit(seed=20260921, trials=100):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(trials):
        states = int(rng.integers(5, 40)); frames = int(rng.integers(2, 30))
        initial = rng.random(states); initial /= initial.sum()
        costs = rng.random((frames, states)); truth = rng.uniform(0, states - 1, frames)
        weights = rng.random(4); weights /= weights.sum()
        rows.append(decompose(initial, costs, truth, .1, weights))
    return rows


def main():
    rows = random_audit()
    max_p = max(row["posterior_residual"] for row in rows)
    max_r = max(row["risk_residual"] for row in rows)
    signs = int(sum(np.sign(row["erasure_effect"]) != np.sign(row["projected_help_sum"]) for row in rows))
    result = {"scope": "algebraic synthetic audit before corpus replay",
              "trials": len(rows), "max_posterior_residual": max_p,
              "max_risk_residual": max_r, "sign_mismatches": signs,
              "passed": bool(max_p <= 1e-12 and max_r <= 1e-12 and signs == 0),
              "natural_temporal_p3_confirmed": False}
    out = ROOT / "results/p3_dc_conditional_erasure"; out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
