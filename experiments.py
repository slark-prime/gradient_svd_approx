import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentConfig:
    m: int = 100
    n: int = 100
    r_true: int = 10
    ranks: Tuple[int, ...] = (1, 2, 5, 10)
    kappas: Tuple[float, ...] = (1, 5, 10, 50, 100)
    trials: int = 20
    alpha: float = 0.7
    noise_std: float = 0.01
    trust_region_radius: float = 1.0
    seed: int = 0
    two_sided: bool = True
    time_steps: int = 50
    exp4_ranks: Tuple[int, ...] = (5, 10)
    beta: float = 0.95
    epsilon: float = 1e-8
    drift_scale: float = 0.0


def _spd_matrix(dim: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a symmetric positive definite matrix with the given condition number.

    The eigenvalues are spaced geometrically to control anisotropy. This matches
    the motivation of testing how non-isotropic metrics change the outcome.
    """

    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    evals = np.geomspace(1.0, kappa, num=dim)
    return q @ np.diag(evals) @ q.T


def _matrix_sqrt_and_inv(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(matrix)
    sqrt_evals = np.sqrt(evals)
    inv_sqrt_evals = 1.0 / sqrt_evals
    sqrt_mat = (evecs * sqrt_evals) @ evecs.T
    inv_sqrt_mat = (evecs * inv_sqrt_evals) @ evecs.T
    return sqrt_mat, inv_sqrt_mat


def _truncated_svd(mat: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    return u[:, :rank], s[:rank], vt[:rank]


def _weighted_error(
    G: np.ndarray,
    approx: np.ndarray,
    W_L: np.ndarray,
    W_R: np.ndarray,
    W_L_sqrt: np.ndarray = None,
    W_R_sqrt: np.ndarray = None,
) -> float:
    diff = G - approx
    if W_L_sqrt is None:
        W_L_sqrt, _ = _matrix_sqrt_and_inv(W_L)
    if W_R_sqrt is None:
        W_R_sqrt, _ = _matrix_sqrt_and_inv(W_R)
    weighted = W_L_sqrt @ diff @ W_R_sqrt
    return np.linalg.norm(weighted, ord="fro")


def _whitened_rank_k(G: np.ndarray, W_L: np.ndarray, W_R: np.ndarray, rank: int) -> np.ndarray:
    W_L_sqrt, W_L_inv_sqrt = _matrix_sqrt_and_inv(W_L)
    W_R_sqrt, W_R_inv_sqrt = _matrix_sqrt_and_inv(W_R)
    G_tilde = W_L_sqrt @ G @ W_R_sqrt
    u_tilde, s_tilde, vt_tilde = _truncated_svd(G_tilde, rank)
    return W_L_inv_sqrt @ (u_tilde @ np.diag(s_tilde) @ vt_tilde) @ W_R_inv_sqrt


def _generate_matrix(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    U, _ = np.linalg.qr(rng.standard_normal((cfg.m, cfg.r_true)))
    V, _ = np.linalg.qr(rng.standard_normal((cfg.n, cfg.r_true)))
    sigmas = np.array([cfg.alpha ** i for i in range(cfg.r_true)])
    core = U @ np.diag(sigmas) @ V.T
    noise = cfg.noise_std * rng.standard_normal((cfg.m, cfg.n))
    return core + noise


def _drift_orthonormal(
    base: np.ndarray, rng: np.random.Generator, scale: float
) -> np.ndarray:
    if scale <= 0:
        return base
    perturbed = base + scale * rng.standard_normal(base.shape)
    q, _ = np.linalg.qr(perturbed)
    return q[:, : base.shape[1]]


def _write_records(records: Iterable[Dict[str, float]], path: Path) -> None:
    rows = list(records)
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_experiment1(cfg: ExperimentConfig, out_dir: Path) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []

    for kappa in cfg.kappas:
        W_L = _spd_matrix(cfg.m, kappa, rng)
        W_R = _spd_matrix(cfg.n, kappa, rng) if cfg.two_sided else W_L
        W_L_sqrt, _ = _matrix_sqrt_and_inv(W_L)
        W_R_sqrt, _ = _matrix_sqrt_and_inv(W_R)
        for trial in range(cfg.trials):
            G = _generate_matrix(cfg, rng)
            for k in cfg.ranks:
                u, s, vt = _truncated_svd(G, k)
                Gk = u @ np.diag(s) @ vt
                whitened = _whitened_rank_k(G, W_L, W_R, k)
                err_vanilla = _weighted_error(G, Gk, W_L, W_R, W_L_sqrt, W_R_sqrt)
                err_white = _weighted_error(G, whitened, W_L, W_R, W_L_sqrt, W_R_sqrt)
                records.append(
                    {
                        "experiment": 1,
                        "trial": trial,
                        "kappa": kappa,
                        "rank": k,
                        "err_vanilla": err_vanilla,
                        "err_whitened": err_white,
                    }
                )

    for k in cfg.ranks:
        plt.figure(figsize=(6, 4))
        for method in ["err_vanilla", "err_whitened"]:
            means = []
            stds = []
            for kappa in cfg.kappas:
                vals = [r[method] for r in records if r["rank"] == k and r["kappa"] == kappa]
                means.append(np.mean(vals))
                stds.append(np.std(vals) / math.sqrt(len(vals)))
            plt.errorbar(cfg.kappas, means, yerr=stds, label=method.replace("err_", ""))
        plt.xlabel("Condition number κ(W)")
        plt.ylabel("Weighted error |W^{1/2}(G-X)|_F")
        plt.title(f"Experiment 1: weighted error, rank={k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"experiment1_rank{k}.png", dpi=200)
        plt.close()

    _write_records(records, out_dir / "experiment1_records.csv")
    return records


def run_experiment2(cfg: ExperimentConfig, out_dir: Path) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []

    for kappa in cfg.kappas:
        W_L = _spd_matrix(cfg.m, kappa, rng)
        W_R = _spd_matrix(cfg.n, kappa, rng) if cfg.two_sided else W_L
        W_L_sqrt, W_L_inv_sqrt = _matrix_sqrt_and_inv(W_L)
        W_R_sqrt, W_R_inv_sqrt = _matrix_sqrt_and_inv(W_R)
        for trial in range(cfg.trials):
            G = _generate_matrix(cfg, rng)
            for k in cfg.ranks:
                u, s, vt = _truncated_svd(G, k)
                Gk = u @ np.diag(s) @ vt
                D_vanilla = -Gk
                alpha = cfg.trust_region_radius / np.linalg.norm(
                    W_L_sqrt @ D_vanilla @ W_R_sqrt, ord="fro"
                )
                delta_vanilla = alpha * D_vanilla

                G_tilde = W_L_sqrt @ G @ W_R_sqrt
                u_t, s_t, vt_t = _truncated_svd(G_tilde, k)
                G_tilde_k = u_t @ np.diag(s_t) @ vt_t
                beta = cfg.trust_region_radius / np.linalg.norm(G_tilde_k, ord="fro")
                delta_white = -beta * (W_L_inv_sqrt @ G_tilde_k @ W_R_inv_sqrt)

                drop_vanilla = -np.tensordot(G, delta_vanilla, axes=2)
                drop_white = -np.tensordot(G, delta_white, axes=2)

                records.append(
                    {
                        "experiment": 2,
                        "trial": trial,
                        "kappa": kappa,
                        "rank": k,
                        "drop_vanilla": float(drop_vanilla),
                        "drop_whitened": float(drop_white),
                    }
                )

    for k in cfg.ranks:
        plt.figure(figsize=(6, 4))
        for method in ["drop_vanilla", "drop_whitened"]:
            means = []
            stds = []
            for kappa in cfg.kappas:
                vals = [r[method] for r in records if r["rank"] == k and r["kappa"] == kappa]
                means.append(np.mean(vals))
                stds.append(np.std(vals) / math.sqrt(len(vals)))
            plt.errorbar(cfg.kappas, means, yerr=stds, label=method.replace("drop_", ""))
        plt.xlabel("Condition number κ(W)")
        plt.ylabel("Linearized loss drop")
        plt.title(f"Experiment 2: trust-region drop, rank={k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"experiment2_rank{k}.png", dpi=200)
        plt.close()

    _write_records(records, out_dir / "experiment2_records.csv")
    return records


def run_experiment3(out_dir: Path) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    G = np.array([[1.0, 0.2], [2.0, -0.1]])
    kappa = 50.0
    W_L = np.diag([1.0, kappa])
    W_R = np.diag([kappa, 1.0])
    W_L_sqrt, W_L_inv_sqrt = _matrix_sqrt_and_inv(W_L)
    W_R_sqrt, W_R_inv_sqrt = _matrix_sqrt_and_inv(W_R)

    u, s, vt = _truncated_svd(G, 1)
    G1 = u @ np.diag(s) @ vt
    err_vanilla = _weighted_error(G, G1, W_L, W_R, W_L_sqrt, W_R_sqrt)

    G_tilde = W_L_sqrt @ G @ W_R_sqrt
    u_t, s_t, vt_t = _truncated_svd(G_tilde, 1)
    G_tilde1 = u_t @ np.diag(s_t) @ vt_t
    X_star = W_L_inv_sqrt @ G_tilde1 @ W_R_inv_sqrt
    err_white = _weighted_error(G, X_star, W_L, W_R, W_L_sqrt, W_R_sqrt)

    table = {
        "G": G,
        "vanilla": G1,
        "whitened": X_star,
        "err_vanilla": float(err_vanilla),
        "err_whitened": float(err_white),
        "kappa": kappa,
    }

    with open(out_dir / "experiment3_table.txt", "w") as f:
        f.write("Tiny 2x2 counterexample (kappa=50)\n")
        f.write(f"Original G:\n{G}\n\n")
        f.write(f"Rank-1 SVD:\n{G1}\nWeighted error: {err_vanilla:.6f}\n\n")
        f.write(f"Whitened rank-1 SVD:\n{X_star}\nWeighted error: {err_white:.6f}\n")

    return table


def run_experiment4(cfg: ExperimentConfig, out_dir: Path) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed + 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []

    U, _ = np.linalg.qr(rng.standard_normal((cfg.m, cfg.r_true)))
    V, _ = np.linalg.qr(rng.standard_normal((cfg.n, cfg.r_true)))
    sigmas = np.array([cfg.alpha ** i for i in range(cfg.r_true)])

    second_moment = np.zeros(cfg.m * cfg.n)
    cumulative_vanilla = {k: 0.0 for k in cfg.exp4_ranks}
    cumulative_white = {k: 0.0 for k in cfg.exp4_ranks}

    for step in range(cfg.time_steps):
        if step > 0 and cfg.drift_scale > 0:
            U = _drift_orthonormal(U, rng, cfg.drift_scale)
            V = _drift_orthonormal(V, rng, cfg.drift_scale)

        core = U @ np.diag(sigmas) @ V.T
        noise = cfg.noise_std * rng.standard_normal((cfg.m, cfg.n))
        G = core + noise

        g_vec = G.reshape(-1)
        squared = g_vec * g_vec
        second_moment = cfg.beta * second_moment + (1.0 - cfg.beta) * squared
        w_diag = 1.0 / (np.sqrt(second_moment) + cfg.epsilon)
        sqrt_w = np.sqrt(w_diag).reshape(cfg.m, cfg.n)
        inv_sqrt_w = 1.0 / sqrt_w

        for k in cfg.exp4_ranks:
            u, s, vt = _truncated_svd(G, k)
            Gk = u @ np.diag(s) @ vt
            D_vanilla = -Gk
            weighted_norm = np.linalg.norm(sqrt_w * D_vanilla, ord="fro")
            alpha = cfg.trust_region_radius / weighted_norm if weighted_norm > 0 else 0.0
            delta_vanilla = alpha * D_vanilla

            drop_vanilla = -np.tensordot(G, delta_vanilla, axes=2)

            G_tilde = sqrt_w * G
            u_t, s_t, vt_t = _truncated_svd(G_tilde, k)
            G_tilde_k = u_t @ np.diag(s_t) @ vt_t
            tilde_norm = np.linalg.norm(G_tilde_k, ord="fro")
            beta_scale = (
                cfg.trust_region_radius / tilde_norm if tilde_norm > 0 else 0.0
            )
            delta_white = -beta_scale * (G_tilde_k * inv_sqrt_w)
            drop_white = -np.tensordot(G, delta_white, axes=2)

            cumulative_vanilla[k] += float(drop_vanilla)
            cumulative_white[k] += float(drop_white)

            records.append(
                {
                    "experiment": 4,
                    "step": step,
                    "rank": k,
                    "drop_vanilla": float(drop_vanilla),
                    "drop_whitened": float(drop_white),
                    "cum_drop_vanilla": cumulative_vanilla[k],
                    "cum_drop_whitened": cumulative_white[k],
                    "beta": cfg.beta,
                    "epsilon": cfg.epsilon,
                    "drift_scale": cfg.drift_scale,
                }
            )

    for k in cfg.exp4_ranks:
        steps = [r["step"] for r in records if r["rank"] == k]
        cum_v = [r["cum_drop_vanilla"] for r in records if r["rank"] == k]
        cum_w = [r["cum_drop_whitened"] for r in records if r["rank"] == k]

        plt.figure(figsize=(6, 4))
        plt.plot(steps, cum_v, label="vanilla")
        plt.plot(steps, cum_w, label="whitened")
        plt.xlabel("Step")
        plt.ylabel("Cumulative linearized drop")
        plt.title(f"Experiment 4: adaptive metric, rank={k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"experiment4_rank{k}.png", dpi=200)
        plt.close()

    summary_rows = []
    for k in cfg.exp4_ranks:
        ratio = cumulative_white[k] / cumulative_vanilla[k] if cumulative_vanilla[k] else float("inf")
        summary_rows.append(
            {
                "rank": k,
                "cum_drop_vanilla": cumulative_vanilla[k],
                "cum_drop_whitened": cumulative_white[k],
                "ratio_whitened_over_vanilla": ratio,
                "beta": cfg.beta,
                "epsilon": cfg.epsilon,
                "drift_scale": cfg.drift_scale,
                "time_steps": cfg.time_steps,
            }
        )

    _write_records(records, out_dir / "experiment4_records.csv")
    _write_records(summary_rows, out_dir / "experiment4_summary.csv")
    return records


def main():
    parser = argparse.ArgumentParser(description="Run weighted SVD experiments.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for outputs")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=(),
        choices=["exp1", "exp2", "exp3", "exp4"],
        help="Experiments to skip",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    out_dir = args.output
    out_dir.mkdir(exist_ok=True)

    if "exp1" not in args.skip:
        run_experiment1(cfg, out_dir / "exp1")
    if "exp2" not in args.skip:
        run_experiment2(cfg, out_dir / "exp2")
    if "exp3" not in args.skip:
        run_experiment3(out_dir / "exp3")
    if "exp4" not in args.skip:
        run_experiment4(cfg, out_dir / "exp4")


if __name__ == "__main__":
    main()
