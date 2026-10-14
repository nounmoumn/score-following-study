#!/usr/bin/env python3
"""Numerically audit the exact P3-E prediction-only recurrence."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p3_e_formula_audit/result.json"
TRANSITION = np.asarray([0.2, 0.65, 0.1, 0.05])
RESTART = 1e-6
SEED = 20260921


def transition_matrix(n, weights=TRANSITION):
    matrix = np.zeros((n, n))
    for source in range(n):
        for step, weight in enumerate(weights):
            destination = min(source + step, n - 1)
            matrix[destination, source] += weight
    return matrix


def update(posterior, cost, matrix, beta=8.0, restart=RESTART):
    n = len(posterior)
    prior = (1 - restart) * (matrix @ posterior) + restart / n
    likelihood = np.exp(-beta * (cost - np.min(cost)))
    result = prior * likelihood
    return result / result.sum(), prior


def closed_form(initial, matrix, frames, restart=RESTART):
    n = len(initial)
    a = (1 - restart) * matrix
    total = np.linalg.matrix_power(a, frames) @ initial
    uniform = np.full(n, 1 / n)
    for power in range(frames):
        total += restart * (np.linalg.matrix_power(a, power) @ uniform)
    return total


def audit(seed=SEED, n=37, frames=23):
    rng = np.random.default_rng(seed)
    matrix = transition_matrix(n)
    initial = rng.random(n)
    initial /= initial.sum()
    posterior = initial.copy()
    max_erasure_update_residual = 0.0
    max_sham_update_residual = 0.0
    for _ in range(frames):
        cost = rng.random(n)
        intact, _ = update(posterior, cost, matrix)
        sham, _ = update(posterior, cost + float(cost.mean()), matrix)
        erased, prior = update(posterior, np.full(n, float(cost.mean())), matrix)
        max_erasure_update_residual = max(max_erasure_update_residual,
                                           float(np.max(np.abs(erased - prior))))
        max_sham_update_residual = max(max_sham_update_residual,
                                        float(np.max(np.abs(sham - intact))))
        posterior = erased
    formula = closed_form(initial, matrix, frames)
    result = {
        "states": n, "frames": frames, "seed": seed,
        "transition_column_sum_residual": float(np.max(np.abs(matrix.sum(axis=0) - 1))),
        "erasure_posterior_prior_residual": max_erasure_update_residual,
        "row_constant_sham_residual": max_sham_update_residual,
        "closed_form_residual": float(np.max(np.abs(posterior - formula))),
    }
    result["all_exact_within_1e_12"] = all(value <= 1e-12 for key, value in result.items()
                                               if key.endswith("residual"))
    return result


def main():
    result = audit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

