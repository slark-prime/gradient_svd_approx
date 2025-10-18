"""Linear algebra helper routines used across experiments."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def truncated_svd(matrix: ArrayLike, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the leading rank singular triplets of a matrix."""

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    u_k = u[:, :rank]
    s_k = s[:rank]
    vt_k = vt[:rank, :]
    return u_k, s_k, vt_k


def svd_reconstruct(u: np.ndarray, s: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Reconstruct a matrix from a truncated SVD."""

    return (u * s) @ vt


def randomized_svd(matrix: ArrayLike, rank: int, oversample: int = 5, n_iter: int = 2,
                   rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a randomized SVD approximation following Halko et al."""

    if rng is None:
        rng = np.random.default_rng()

    a = np.asarray(matrix)
    m, n = a.shape
    l = min(rank + oversample, min(m, n))
    omega = rng.standard_normal((n, l))
    y = a @ omega
    for _ in range(n_iter):
        y = a @ (a.T @ y)
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ a
    u_tilde, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ u_tilde
    return u[:, :rank], s[:rank], vt[:rank, :]


def cur_decomposition(matrix: ArrayLike, rank: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """A simple CUR approximation using leverage-score style sampling."""

    if rng is None:
        rng = np.random.default_rng()

    a = np.asarray(matrix)
    m, n = a.shape
    u, s, vt = truncated_svd(a, rank)
    leverage_rows = np.sum(u ** 2, axis=1)
    leverage_cols = np.sum(vt.T ** 2, axis=1)
    row_probs = leverage_rows / leverage_rows.sum()
    col_probs = leverage_cols / leverage_cols.sum()
    row_idx = rng.choice(m, size=rank, p=row_probs, replace=True)
    col_idx = rng.choice(n, size=rank, p=col_probs, replace=True)
    scaling_rows = np.sqrt(rank * row_probs[row_idx])
    scaling_cols = np.sqrt(rank * col_probs[col_idx])
    c = a[:, col_idx] / scaling_cols
    r = (a[row_idx, :] / scaling_rows[:, None])
    w = a[np.ix_(row_idx, col_idx)] / (scaling_rows[:, None] * scaling_cols[None, :])
    u_matrix = np.linalg.pinv(w)
    return c @ u_matrix @ r


def whitened_svd(matrix: ArrayLike, w_left: np.ndarray, w_right: np.ndarray, rank: int) -> np.ndarray:
    """Perform weighted low-rank approximation via congruence transformation."""

    w_left_sqrt = _matrix_sqrt(w_left)
    w_right_sqrt = _matrix_sqrt(w_right)
    transformed = w_left_sqrt @ matrix @ w_right_sqrt
    u, s, vt = truncated_svd(transformed, rank)
    w_left_inv_sqrt = np.linalg.inv(w_left_sqrt)
    w_right_inv_sqrt = np.linalg.inv(w_right_sqrt)
    approx = w_left_inv_sqrt @ (u * s) @ vt @ w_right_inv_sqrt
    return approx


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if np.any(eigvals <= 0):
        raise ValueError("Matrix square root requires positive definite input")
    sqrt = eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.T)
    return sqrt


def weighted_frobenius_error(matrix: np.ndarray, approx: np.ndarray,
                              w_left: np.ndarray, w_right: np.ndarray) -> float:
    diff = matrix - approx
    w_left_sqrt = _matrix_sqrt(w_left)
    w_right_sqrt = _matrix_sqrt(w_right)
    weighted = w_left_sqrt @ diff @ w_right_sqrt
    return float(np.linalg.norm(weighted, ord="fro"))


def frobenius_error(matrix: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - approx, ord="fro"))


def spectral_error(matrix: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(matrix - approx, ord=2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = a.ravel()
    b_vec = b.ravel()
    numerator = float(np.dot(a_vec, b_vec))
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0:
        return 0.0
    return numerator / denom


def weighted_inner_product(a: np.ndarray, b: np.ndarray,
                           w_left: np.ndarray, w_right: np.ndarray) -> float:
    """Compute the weighted Frobenius inner product under SPD weights."""

    w_left_sqrt = _matrix_sqrt(w_left)
    w_right_sqrt = _matrix_sqrt(w_right)
    a_weighted = w_left_sqrt @ a @ w_right_sqrt
    b_weighted = w_left_sqrt @ b @ w_right_sqrt
    return float(np.dot(a_weighted.ravel(), b_weighted.ravel()))


def weighted_cosine_similarity(a: np.ndarray, b: np.ndarray,
                               w_left: np.ndarray, w_right: np.ndarray) -> float:
    """Cosine similarity under the weighted Frobenius inner product."""

    numerator = weighted_inner_product(a, b, w_left, w_right)
    denom_a = weighted_inner_product(a, a, w_left, w_right)
    denom_b = weighted_inner_product(b, b, w_left, w_right)
    denom = np.sqrt(denom_a * denom_b)
    if denom == 0:
        return 0.0
    return numerator / denom
