"""Data generation utilities for gradient SVD experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LowRankMatrixParams:
    """Parameters describing a noisy low-rank matrix."""

    m: int
    n: int
    rank: int
    singular_values: Tuple[float, ...]
    noise_level: float = 0.0
    outlier_density: float = 0.0
    outlier_magnitude: float = 0.0


def make_low_rank_matrix(params: LowRankMatrixParams, rng: np.random.Generator) -> np.ndarray:
    """Generate a low-rank matrix with optional Gaussian noise and sparse outliers."""

    if len(params.singular_values) != params.rank:
        raise ValueError("Number of singular values must match target rank")

    u, _ = np.linalg.qr(rng.standard_normal((params.m, params.rank)))
    v, _ = np.linalg.qr(rng.standard_normal((params.n, params.rank)))
    sigma = np.diag(np.array(params.singular_values, dtype=float))
    base = u @ sigma @ v.T

    if params.noise_level > 0:
        base = base + params.noise_level * rng.standard_normal((params.m, params.n))

    if params.outlier_density > 0 and params.outlier_magnitude > 0:
        mask = rng.random((params.m, params.n)) < params.outlier_density
        signs = np.where(rng.random((params.m, params.n)) < 0.5, -1.0, 1.0)
        outliers = params.outlier_magnitude * signs * mask
        base = base + outliers

    return base


def random_spd_matrix(size: int, condition_number: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive definite matrix with the desired condition number."""

    if condition_number < 1:
        raise ValueError("condition_number must be >= 1")

    q, _ = np.linalg.qr(rng.standard_normal((size, size)))
    # Log-linear spectrum to avoid extremely tiny singular values when kappa is large.
    log_eigs = np.linspace(0.0, np.log(condition_number), size)
    eigs = np.exp(log_eigs)
    return q @ np.diag(eigs) @ q.T


def diagonal_preconditioner(size: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Create a diagonal SPD preconditioner with entries sampled log-uniformly in [low, high]."""

    if low <= 0 or high <= 0:
        raise ValueError("Bounds must be positive")

    log_low, log_high = np.log(low), np.log(high)
    entries = np.exp(rng.uniform(log_low, log_high, size=size))
    return np.diag(entries)


def make_tensor(shape: Tuple[int, int, int], ranks: Tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic 3-way tensor with specified multilinear ranks."""

    if len(shape) != 3 or len(ranks) != 3:
        raise ValueError("Only third-order tensors are supported")

    core = rng.standard_normal(ranks)
    a = rng.standard_normal((shape[0], ranks[0]))
    b = rng.standard_normal((shape[1], ranks[1]))
    c = rng.standard_normal((shape[2], ranks[2]))
    tensor = np.einsum("ia,jb,kc,abc->ijk", a, b, c, core)
    return tensor
