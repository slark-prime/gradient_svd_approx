"""Experiment suite implementing the empirical checks outlined in the README."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import (
    LowRankMatrixParams,
    diagonal_preconditioner,
    make_low_rank_matrix,
    random_spd_matrix,
)
from .linalg_utils import (
    cosine_similarity,
    cur_decomposition,
    frobenius_error,
    randomized_svd,
    spectral_error,
    svd_reconstruct,
    truncated_svd,
    weighted_cosine_similarity,
    weighted_frobenius_error,
    weighted_inner_product,
    whitened_svd,
)
from .robust import trimmed_svd_low_rank
from .streaming import FrequentDirections
from .tensors import hosvd, multilinear_relative_error, unfold


@dataclass
class ExperimentResult:
    name: str
    summary: Dict[str, float | int | str]
    curves: Dict[str, List[float]] | None = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"name": self.name, "summary": self.summary}
        if self.curves is not None:
            payload["curves"] = self.curves
        return payload


def experiment_svd_optimality(rng: np.random.Generator) -> ExperimentResult:
    params = LowRankMatrixParams(m=80, n=60, rank=10, singular_values=tuple(np.linspace(5, 0.5, 10)), noise_level=1e-3)
    matrix = make_low_rank_matrix(params, rng)
    rank = 5
    u, s, vt = truncated_svd(matrix, rank)
    svd_approx = svd_reconstruct(u, s, vt)
    rand_u, rand_s, rand_vt = randomized_svd(matrix, rank, rng=rng)
    rand_approx = svd_reconstruct(rand_u, rand_s, rand_vt)
    cur_approx = cur_decomposition(matrix, rank, rng=rng)

    return ExperimentResult(
        name="svd_optimality",
        summary={
            "fro_svd": frobenius_error(matrix, svd_approx),
            "fro_randomized": frobenius_error(matrix, rand_approx),
            "fro_cur": frobenius_error(matrix, cur_approx),
            "spectral_svd": spectral_error(matrix, svd_approx),
            "spectral_randomized": spectral_error(matrix, rand_approx),
            "spectral_cur": spectral_error(matrix, cur_approx),
        },
    )


def experiment_weighted_geometry(rng: np.random.Generator) -> ExperimentResult:
    params = LowRankMatrixParams(m=50, n=50, rank=8, singular_values=tuple(np.linspace(4, 0.4, 8)))
    matrix = make_low_rank_matrix(params, rng)
    w_left = random_spd_matrix(50, condition_number=120.0, rng=rng)
    w_right = random_spd_matrix(50, condition_number=45.0, rng=rng)
    rank = 5

    u, s, vt = truncated_svd(matrix, rank)
    vanilla = svd_reconstruct(u, s, vt)
    weighted = whitened_svd(matrix, w_left, w_right, rank)

    return ExperimentResult(
        name="weighted_geometry",
        summary={
            "weighted_error_vanilla": weighted_frobenius_error(matrix, vanilla, w_left, w_right),
            "weighted_error_whitened": weighted_frobenius_error(matrix, weighted, w_left, w_right),
            "fro_vanilla": frobenius_error(matrix, vanilla),
            "fro_whitened": frobenius_error(matrix, weighted),
        },
    )


def experiment_descent_alignment(rng: np.random.Generator) -> ExperimentResult:
    params = LowRankMatrixParams(m=60, n=60, rank=6, singular_values=tuple(np.linspace(6, 0.6, 6)))
    matrix = make_low_rank_matrix(params, rng)
    # Diagonal trust-region metric with strong anisotropy.
    weights = diagonal_preconditioner(60, low=0.05, high=15.0, rng=rng)
    w_left = weights @ weights
    w_right = w_left
    rank = 4

    u, s, vt = truncated_svd(matrix, rank)
    vanilla = svd_reconstruct(u, s, vt)
    weighted = whitened_svd(matrix, w_left, w_right, rank)

    vanilla_weighted_cos = weighted_cosine_similarity(matrix, vanilla, w_left, w_right)
    weighted_cos = weighted_cosine_similarity(matrix, weighted, w_left, w_right)
    vanilla_drop = weighted_inner_product(matrix, vanilla, w_left, w_right)
    weighted_drop = weighted_inner_product(matrix, weighted, w_left, w_right)

    return ExperimentResult(
        name="descent_alignment",
        summary={
            "weighted_cosine_vanilla": vanilla_weighted_cos,
            "weighted_cosine_whitened": weighted_cos,
            "weighted_drop_vanilla": vanilla_drop,
            "weighted_drop_whitened": weighted_drop,
            "standard_cosine_vanilla": cosine_similarity(matrix, vanilla),
            "standard_cosine_whitened": cosine_similarity(matrix, weighted),
        },
    )


def experiment_robustness(rng: np.random.Generator) -> ExperimentResult:
    m = n = 80
    rank = 5
    singular_values = tuple(np.geomspace(6.0, 0.8, rank))
    u, _ = np.linalg.qr(rng.standard_normal((m, rank)))
    v, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    sigma = np.diag(np.array(singular_values))
    clean = u @ sigma @ v.T

    matrix = clean.copy()
    matrix += 0.5 * rng.standard_normal((m, n))
    outlier_mask = rng.random((m, n)) < 0.08
    outlier_signs = np.where(rng.random((m, n)) < 0.5, -1.0, 1.0)
    matrix += 25.0 * outlier_mask * outlier_signs

    u, s, vt = truncated_svd(matrix, rank)
    vanilla = svd_reconstruct(u, s, vt)
    robust = trimmed_svd_low_rank(matrix, rank, quantile=0.92)

    l1_clean_vanilla = float(np.linalg.norm(clean - vanilla, ord=1))
    l1_clean_robust = float(np.linalg.norm(clean - robust, ord=1))
    fro_clean_vanilla = frobenius_error(clean, vanilla)
    fro_clean_robust = frobenius_error(clean, robust)

    return ExperimentResult(
        name="robustness",
        summary={
            "l1_to_clean_vanilla": l1_clean_vanilla,
            "l1_to_clean_trimmed": l1_clean_robust,
            "fro_to_clean_vanilla": fro_clean_vanilla,
            "fro_to_clean_trimmed": fro_clean_robust,
            "cosine_to_clean_vanilla": cosine_similarity(clean, vanilla),
            "cosine_to_clean_trimmed": cosine_similarity(clean, robust),
        },
    )


def experiment_tensor(rng: np.random.Generator) -> ExperimentResult:
    shape = (32, 18, 12)
    clean_ranks = (6, 5, 4)
    target_ranks = (4, 3, 3)

    grid0 = np.arange(clean_ranks[0])[:, None, None]
    grid1 = np.arange(clean_ranks[1])[None, :, None]
    grid2 = np.arange(clean_ranks[2])[None, None, :]
    core = np.exp(-0.8 * grid0 - 0.3 * grid1 - 0.2 * grid2)
    core += 0.01 * rng.standard_normal(clean_ranks)
    factors = []
    for dim, rank in zip(shape, clean_ranks):
        factor, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
        factors.append(factor)
    clean_tensor = np.einsum("ia,jb,kc,abc->ijk", factors[0], factors[1], factors[2], core)

    # Low matrix-rank interference that breaks multilinear structure but dominates
    # the matricized SVD objective when the tensor is flattened.
    interference_left = rng.standard_normal((shape[0], 2))
    interference_right = rng.standard_normal((2, shape[1] * shape[2]))
    interference = 1.6 * (interference_left @ interference_right).reshape(shape)
    observed = clean_tensor + interference

    unfolding = unfold(observed, 0)
    u, s, vt = truncated_svd(unfolding, target_ranks[0])
    matricized_tensor = (u * s) @ vt
    matricized_tensor = matricized_tensor.reshape(observed.shape)
    hosvd_approx = hosvd(observed, target_ranks)

    def _mode_basis(t: np.ndarray, mode: int, r: int) -> np.ndarray:
        unfolding = unfold(t, mode)
        u_mode, _, _ = np.linalg.svd(unfolding, full_matrices=False)
        return u_mode[:, :r]

    def _avg_principal_angle(true_basis: np.ndarray, approx_basis: np.ndarray) -> float:
        _, s_vals, _ = np.linalg.svd(true_basis.T @ approx_basis, full_matrices=False)
        cosines = np.clip(s_vals, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        return float(np.mean(angles))

    true_bases = [factor[:, :target] for factor, target in zip(factors, target_ranks)]
    matricized_bases = [_mode_basis(matricized_tensor, mode, target_ranks[mode]) for mode in range(3)]
    hosvd_bases = [_mode_basis(hosvd_approx, mode, target_ranks[mode]) for mode in range(3)]

    return ExperimentResult(
        name="tensor",
        summary={
            "relative_error_to_clean_matricized": multilinear_relative_error(
                clean_tensor, matricized_tensor
            ),
            "relative_error_to_clean_hosvd": multilinear_relative_error(clean_tensor, hosvd_approx),
            "relative_error_to_observed_matricized": multilinear_relative_error(
                observed, matricized_tensor
            ),
            "relative_error_to_observed_hosvd": multilinear_relative_error(observed, hosvd_approx),
            "avg_angle_deg_matricized": float(np.mean([
                _avg_principal_angle(true, approx)
                for true, approx in zip(true_bases, matricized_bases)
            ])),
            "avg_angle_deg_hosvd": float(np.mean([
                _avg_principal_angle(true, approx)
                for true, approx in zip(true_bases, hosvd_bases)
            ])),
        },
    )


def experiment_streaming(rng: np.random.Generator) -> ExperimentResult:
    m, n = 40, 30
    rank = 4
    n_steps = 60
    refresh_interval = 8

    fd = FrequentDirections(m * n, rank)
    cumulative_fd_error = 0.0
    cumulative_static_error = 0.0
    cumulative_periodic_error = 0.0
    cumulative_optimal_error = 0.0

    left_basis = rng.standard_normal((m, rank))
    right_basis = rng.standard_normal((n, rank))
    singular_values = np.linspace(5, 1, rank)
    static_u = None
    static_vt = None
    periodic_u = None
    periodic_s = None
    periodic_vt = None
    periodic_recomputes = 0

    for step in range(n_steps):
        left_basis = 0.98 * left_basis + 0.02 * rng.standard_normal((m, rank))
        right_basis = 0.98 * right_basis + 0.02 * rng.standard_normal((n, rank))
        gradient = left_basis @ (singular_values[:, None] * right_basis.T)
        gradient += 0.02 * rng.standard_normal((m, n))

        row = gradient.reshape(-1)
        fd.update(row)
        fd_proj = fd.project(row).reshape(m, n)
        cumulative_fd_error += frobenius_error(gradient, fd_proj)

        u, s, vt = truncated_svd(gradient, rank)
        optimal_step = svd_reconstruct(u, s, vt)
        cumulative_optimal_error += frobenius_error(gradient, optimal_step)

        if static_u is None:
            static_u = u
            static_vt = vt

        coeff = static_u.T @ gradient @ static_vt.T
        static_proj = static_u @ coeff @ static_vt
        cumulative_static_error += frobenius_error(gradient, static_proj)

        if periodic_u is None or step % refresh_interval == 0:
            periodic_u, periodic_s, periodic_vt = u, s, vt
            periodic_recomputes += 1

        periodic_step = svd_reconstruct(periodic_u, periodic_s, periodic_vt)
        cumulative_periodic_error += frobenius_error(gradient, periodic_step)

    return ExperimentResult(
        name="streaming",
        summary={
            "cumulative_error_fd": cumulative_fd_error,
            "cumulative_error_static": cumulative_static_error,
            "cumulative_error_periodic_svd": cumulative_periodic_error,
            "cumulative_error_optimal_per_step": cumulative_optimal_error,
            "fd_shrink_calls": fd.num_shrinks,
            "periodic_svd_recomputes": periodic_recomputes,
        },
    )


def run_all(seed: int = 1234, output_dir: str | Path = "experiments/results") -> List[ExperimentResult]:
    rng = np.random.default_rng(seed)
    results = [
        experiment_svd_optimality(rng),
        experiment_weighted_geometry(rng),
        experiment_descent_alignment(rng),
        experiment_robustness(rng),
        experiment_tensor(rng),
        experiment_streaming(rng),
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    serialized = [result.to_dict() for result in results]
    np.save(output_path / "summary.npy", serialized, allow_pickle=True)
    with open(output_path / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(format_results(results))
    return results


def format_results(results: List[ExperimentResult]) -> str:
    lines: List[str] = []
    for result in results:
        lines.append(f"## {result.name}")
        for key, value in result.summary.items():
            lines.append(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_all()
    print(format_results(results))
