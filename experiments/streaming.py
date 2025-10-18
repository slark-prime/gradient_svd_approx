"""Streaming low-rank maintenance algorithms."""
from __future__ import annotations

import numpy as np


class FrequentDirections:
    """Compact sketch for streaming matrices following Liberty (2013)."""

    def __init__(self, dimension: int, rank: int):
        self.dimension = dimension
        self.rank = rank
        self.ell = 2 * rank
        self.basis = np.zeros((self.ell, dimension))
        self.n_rows = 0
        self.num_shrinks = 0

    def _shrink(self) -> None:
        u, s, vt = np.linalg.svd(self.basis, full_matrices=False)
        delta = s[self.rank - 1] ** 2 if self.rank > 0 else 0.0
        shrinked = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        self.basis = np.diag(shrinked) @ vt
        self.num_shrinks += 1

    def update(self, row: np.ndarray) -> None:
        if row.shape != (self.dimension,):
            raise ValueError("Row dimension mismatch")
        if self.n_rows < self.ell:
            self.basis[self.n_rows, :] = row
            self.n_rows += 1
        else:
            self._shrink()
            self.basis[self.rank, :] = row
            self.n_rows = self.rank + 1

    def get_basis(self) -> np.ndarray:
        if self.n_rows == 0:
            return np.zeros((self.rank, self.dimension))
        u, _, vt = np.linalg.svd(self.basis[: self.n_rows, :], full_matrices=False)
        return vt[: self.rank, :]

    def project(self, row: np.ndarray) -> np.ndarray:
        basis = self.get_basis()
        return basis.T @ (basis @ row)
