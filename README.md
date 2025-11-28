# Gradient SVD Approximation

This repository contains a small collection of scripts for running synthetic experiments on weighted singular value decomposition (SVD) and generating publication-quality plots.

## File structure

- `experiments.py`: Driver for four synthetic experiments (error comparisons, trust-region drops, a 2×2 counterexample, and an adaptive metric study) that write metrics and figures into the `results/` directory.
- `make_paper_plots.py`: Utilities for re-plotting experiment outputs into polished figures such as weighted error vs. rank, relative improvement vs. condition number, drop ratios, and adaptive-run summaries.
- `paper_figures/`: Example outputs prepared for a paper, typically produced by `make_paper_plots.py`.
- `results/`: Default output directory populated by `experiments.py` (CSV logs and PNGs).
- `requirements.txt`: Python dependencies for running experiments and plotting.

## Scripts

### experiments.py

The `experiments.py` script runs four experiments that explore how preconditioning ("whitening") affects low-rank approximations and trust-region updates:

1. **Experiment 1:** Compares weighted Frobenius errors between vanilla truncated SVD and whitened SVD across varying condition numbers and target ranks.
2. **Experiment 2:** Measures the linearized loss drop from a single trust-region step using vanilla vs. whitened updates.
3. **Experiment 3:** Demonstrates a 2×2 counterexample where the whitened rank-1 approximation achieves a smaller weighted error than the vanilla SVD solution.
4. **Experiment 4:** Simulates an adaptive diagonal metric with momentum-like updates, tracking cumulative drops for multiple ranks over time.

Run all experiments (writing to `results/`) with:

```bash
python experiments.py
```

Use `--skip` to omit selected experiments (e.g., `--skip exp3 exp4`) and `--output` to choose a different results directory.

### make_paper_plots.py

The `make_paper_plots.py` script reuses saved CSV logs to generate cleaner plots for reporting:

- Weighted error vs. rank (log-scale ranks) for a specific condition number.
- Relative weighted-error improvement vs. conditioning, across all ranks.
- Ratio of whitened to vanilla trust-region drops as conditioning changes.
- Cumulative adaptive-drop curves for a selected rank.

Example usage on Experiment 1 logs:

```bash
python make_paper_plots.py --records results/exp1/experiment1_records.csv \
  --out paper_figures/exp1_rel_gap.png --exp exp1_gap
```

Use `--help` for the full list of plotting modes and arguments.
