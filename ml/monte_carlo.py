"""Monte Carlo race simulation.

Given per-driver mean (mu) and stddev (sigma) of finishing-position score, draw
``n_sims`` race orderings, sort, and tally how often each driver finishes in
each position. Returns the (n_drivers x n_drivers) probability matrix where
row i = P(driver i finishes in position j+1).

This enforces the constraint that exactly one driver finishes per position
— sorting the sampled scores does that automatically.
"""
from __future__ import annotations

import numpy as np


def run_simulation(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    mu, sigma : 1-D arrays of length n_drivers
    n_sims    : number of Monte Carlo samples
    rng       : optional numpy Generator (deterministic in tests)

    Returns
    -------
    prob_matrix : (n_drivers, n_drivers) array
        prob_matrix[i, k] = P(driver i finishes in position k+1).
        Each row sums to 1.0; each column sums to 1.0.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if mu.shape != sigma.shape or mu.ndim != 1:
        raise ValueError("mu and sigma must be 1-D arrays of equal length")
    n = mu.shape[0]
    if n == 0:
        return np.zeros((0, 0))

    sigma = np.clip(sigma, 1e-6, None)
    rng = rng or np.random.default_rng()

    # samples shape: (n_sims, n_drivers)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_sims, n))
    # argsort along driver axis: row k of order = driver indices sorted ascending by score
    order = np.argsort(samples, axis=1)
    # finish_pos[s, i] = position (0-indexed) driver i finished in sim s
    finish_pos = np.empty_like(order)
    sims_idx = np.arange(n_sims)[:, None]
    finish_pos[sims_idx, order] = np.arange(n)[None, :]

    prob = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        counts = np.bincount(finish_pos[:, i], minlength=n)
        prob[i] = counts / n_sims

    return prob


def derive_scalars(prob_matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Win / podium / points probability and expected position from the matrix."""
    n = prob_matrix.shape[0]
    positions = np.arange(1, n + 1, dtype=np.float64)
    return {
        "expected_position": prob_matrix @ positions,
        "win_probability": prob_matrix[:, 0],
        "podium_probability": prob_matrix[:, :3].sum(axis=1),
        "points_probability": prob_matrix[:, :10].sum(axis=1),
    }


# ---------------------------------------------------------------------------
# Self-test (run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mu = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
    sigma = np.array([1.5, 1.5, 2.0, 2.0, 2.5])
    prob = run_simulation(mu, sigma, n_sims=20_000, rng=np.random.default_rng(42))
    print("row sums:", prob.sum(axis=1))
    print("col sums:", prob.sum(axis=0))
    print("matrix:\n", np.round(prob, 3))
    assert np.allclose(prob.sum(axis=1), 1.0), "row sums broken"
    assert np.allclose(prob.sum(axis=0), 1.0), "col sums broken"
    print("OK")
