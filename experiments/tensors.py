"""Tensor helpers for comparing matricized SVD and HOSVD."""
from __future__ import annotations

import numpy as np


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def _mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    tensor = np.moveaxis(tensor, mode, 0)
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    result = matrix @ tensor
    new_shape = (matrix.shape[0],) + shape[1:]
    result = result.reshape(new_shape)
    return np.moveaxis(result, 0, mode)


def hosvd(tensor: np.ndarray, ranks: tuple[int, int, int]) -> np.ndarray:
    modes = tensor.ndim
    if modes != 3:
        raise ValueError("Only third-order tensors supported")
    factors = []
    for mode in range(modes):
        unfolding = unfold(tensor, mode)
        u, _, _ = np.linalg.svd(unfolding, full_matrices=False)
        factors.append(u[:, : ranks[mode]])

    core = tensor.copy()
    for mode in range(modes):
        core = _mode_n_product(core, factors[mode].T, mode)
    approx = core
    for mode in range(modes):
        approx = _mode_n_product(approx, factors[mode], mode)
    return approx


def multilinear_relative_error(tensor: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(tensor - approx) / np.linalg.norm(tensor))
