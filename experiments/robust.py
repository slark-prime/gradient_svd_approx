"""Robust low-rank solvers used in experiments."""
from __future__ import annotations

import numpy as np

from .linalg_utils import truncated_svd, svd_reconstruct


def trimmed_svd_low_rank(matrix: np.ndarray, rank: int, quantile: float = 0.98) -> np.ndarray:
    """Clip extreme entries before taking a truncated SVD to mitigate sparse outliers."""

    threshold = np.quantile(np.abs(matrix), quantile)
    clipped = np.clip(matrix, -threshold, threshold)
    u, s, vt = truncated_svd(clipped, rank)
    return svd_reconstruct(u, s, vt)
