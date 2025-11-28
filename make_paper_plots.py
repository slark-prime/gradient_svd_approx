import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_records(path: Path) -> List[Dict]:
    """Load a CSV file into a list of dicts with numeric fields converted."""
    records: List[Dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec: Dict = {}
            for k, v in row.items():
                if v is None:
                    rec[k] = v
                    continue
                try:
                    rec[k] = int(v)
                except ValueError:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
            records.append(rec)
    return records


# -------------------- EXPERIMENT 1 --------------------


def plot_exp1_error_vs_rank_log2(
    records: List[Dict],
    kappa: float,
    out_path: Path,
) -> None:
    """
    Exp1: weighted error vs rank, but x-axis is exponent in k = 2^e.

    Uses ALL ranks available at this kappa.
    """
    ranks = sorted({int(r["rank"]) for r in records if float(r["kappa"]) == kappa})
    if not ranks:
        raise ValueError(f"No records found for kappa={kappa} in exp1.")
    exps = [np.log2(k) for k in ranks]

    err_v_means, err_v_stds = [], []
    err_w_means, err_w_stds = [], []

    for k in ranks:
        vals_v = [
            float(r["err_vanilla"])
            for r in records
            if int(r["rank"]) == k and float(r["kappa"]) == kappa
        ]
        vals_w = [
            float(r["err_whitened"])
            for r in records
            if int(r["rank"]) == k and float(r["kappa"]) == kappa
        ]
        vals_v = np.array(vals_v)
        vals_w = np.array(vals_w)
        err_v_means.append(vals_v.mean())
        err_v_stds.append(vals_v.std() / np.sqrt(len(vals_v)))
        err_w_means.append(vals_w.mean())
        err_w_stds.append(vals_w.std() / np.sqrt(len(vals_w)))

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        exps, err_v_means, yerr=err_v_stds, marker="o", label="vanilla", capsize=3
    )
    plt.errorbar(
        exps, err_w_means, yerr=err_w_stds, marker="s", label="whitened", capsize=3
    )
    plt.xlabel(r"Exponent $e$ in $k = 2^e$")
    plt.ylabel(r"Weighted error $\|W_L^{1/2}(G-X)W_R^{1/2}\|_F$")
    plt.title(f"Exp. 1: weighted error vs rank (κ = {kappa:g})")
    plt.xticks(exps, [f"{int(e)}" for e in exps])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_exp1_rel_gap_vs_kappa_log(
    records: List[Dict],
    out_path: Path,
) -> None:
    """
    Exp1: relative weighted-error improvement vs kappa, for ALL ranks.

    y = (E_vanilla - E_whitened) / E_whitened
    x = kappa (log10 scale)
    One curve per rank (powers of 2).
    """
    kappas = sorted({float(r["kappa"]) for r in records})
    ranks = sorted({int(r["rank"]) for r in records})

    plt.figure(figsize=(6, 4))
    for k in ranks:
        rel_means = []
        ks_present = []
        for kappa in kappas:
            vals_v = [
                float(r["err_vanilla"])
                for r in records
                if int(r["rank"]) == k and float(r["kappa"]) == kappa
            ]
            vals_w = [
                float(r["err_whitened"])
                for r in records
                if int(r["rank"]) == k and float(r["kappa"]) == kappa
            ]
            if not vals_v:
                continue
            vals_v = np.array(vals_v)
            vals_w = np.array(vals_w)
            rel = (vals_v - vals_w) / vals_w
            rel_means.append(rel.mean())
            ks_present.append(kappa)
        if rel_means:
            plt.semilogx(
                ks_present,
                rel_means,
                marker="o",
                label=f"k = {k}",
            )

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel(r"Condition number $\kappa(W)$ (log scale)")
    plt.ylabel(r"Relative improvement $(E_{\rm van}-E_{\rm white})/E_{\rm white}$")
    plt.title("Exp. 1: relative weighted-error gap vs κ")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(title="rank", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------- EXPERIMENT 2 --------------------


def plot_exp2_ratio_vs_kappa_log(
    records: List[Dict],
    out_path: Path,
) -> None:
    """
    Exp2: ratio of drops vs kappa, for ALL ranks.

    y = (mean drop_whitened) / (mean drop_vanilla)
    x = kappa (log10 scale)
    One curve per rank.

    > 1 : whitened better
    < 1 : vanilla better
    """
    kappas = sorted({float(r["kappa"]) for r in records})
    ranks = sorted({int(r["rank"]) for r in records})

    plt.figure(figsize=(6, 4))
    for k in ranks:
        ratios = []
        ks_present = []
        for kappa in kappas:
            vals_v = [
                float(r["drop_vanilla"])
                for r in records
                if int(r["rank"]) == k and float(r["kappa"]) == kappa
            ]
            vals_w = [
                float(r["drop_whitened"])
                for r in records
                if int(r["rank"]) == k and float(r["kappa"]) == kappa
            ]
            if not vals_v:
                continue
            vals_v = np.array(vals_v)
            vals_w = np.array(vals_w)
            mask = np.abs(vals_v) > 1e-12
            if not mask.any():
                continue
            ratio = (vals_w[mask].mean() / vals_v[mask].mean())
            ratios.append(ratio)
            ks_present.append(kappa)
        if ratios:
            plt.semilogx(
                ks_present,
                ratios,
                marker="o",
                label=f"k = {k}",
            )

    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.xlabel(r"Condition number $\kappa(W)$ (log scale)")
    plt.ylabel(r"Drop ratio $D_{\rm white} / D_{\rm van}$")
    plt.title("Exp. 2: whitened vs vanilla trust-region drop")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(title="rank", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------- EXPERIMENT 4 --------------------


def plot_exp4_cumulative_drop(
    records: List[Dict],
    rank: int,
    out_path: Path,
) -> None:
    """
    Exp4: cumulative linearized drop vs time for a selected rank.
    (Same as before, but keep it for completeness.)
    """
    rec_rank = [r for r in records if int(r["rank"]) == rank]
    if not rec_rank:
        raise ValueError(f"No records found for rank={rank} in exp4.")

    rec_rank = sorted(rec_rank, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rec_rank]
    cum_v = [float(r["cum_drop_vanilla"]) for r in rec_rank]
    cum_w = [float(r["cum_drop_whitened"]) for r in rec_rank]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, cum_v, marker="o", label="vanilla")
    plt.plot(steps, cum_w, marker="s", label="whitened")
    plt.xlabel("Step")
    plt.ylabel("Cumulative linearized drop")
    plt.title(f"Exp. 4: adaptive diagonal metric (rank = {rank})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -------------------- MAIN --------------------


def main():
    parser = argparse.ArgumentParser(
        description="Make paper-ready plots (log2 / log10) from weighted SVD experiments."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (where exp1/exp2/exp4 live)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_figures_log2"),
        help="Directory to save figures",
    )
    parser.add_argument(
        "--exp1-kappa",
        type=float,
        default=None,
        help="κ to use for Exp1 error-vs-rank plot "
             "(default: max κ found in exp1 records)",
    )
    parser.add_argument(
        "--exp4-rank",
        type=int,
        default=None,
        help="Rank for Exp4 cumulative-drop plot "
             "(default: smallest rank in exp4 records)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Exp1 ----------
    exp1_csv = args.results_dir / "exp1" / "experiment1_records.csv"
    exp1_records = _load_records(exp1_csv)
    all_kappas_exp1 = sorted({float(r["kappa"]) for r in exp1_records})
    kappa_for_fig1 = args.exp1_kappa or max(all_kappas_exp1)

    plot_exp1_error_vs_rank_log2(
        exp1_records,
        kappa=kappa_for_fig1,
        out_path=out_dir / f"fig_exp1_error_vs_rank_log2_kappa{kappa_for_fig1:g}.png",
    )
    plot_exp1_rel_gap_vs_kappa_log(
        exp1_records,
        out_path=out_dir / "fig_exp1_rel_gap_vs_kappa_log.png",
    )

    # ---------- Exp2 ----------
    exp2_csv = args.results_dir / "exp2" / "experiment2_records.csv"
    exp2_records = _load_records(exp2_csv)
    plot_exp2_ratio_vs_kappa_log(
        exp2_records,
        out_path=out_dir / "fig_exp2_ratio_vs_kappa_log.png",
    )

    # ---------- Exp4 ----------
    exp4_csv = args.results_dir / "exp4" / "experiment4_records.csv"
    exp4_records = _load_records(exp4_csv)
    all_ranks_exp4 = sorted({int(r["rank"]) for r in exp4_records})
    rank_for_fig4 = args.exp4_rank or all_ranks_exp4[2]

    plot_exp4_cumulative_drop(
        exp4_records,
        rank=rank_for_fig4,
        out_path=out_dir / f"fig_exp4_cum_drop_rank{rank_for_fig4}.png",
    )

    print("Paper-ready log2/log10 figures saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
