# When Is Truncated SVD the Best Low-Rank Gradient? Optimality, Counterexamples, and Geometry-Aware Alternatives

## 1) Problem Statement & Objectives

Given a matrix-shaped “gradient” (G\in\mathbb{R}^{m\times n}) (standing in for any large Jacobian/weight-gradient/aggregation), study whether the **truncated SVD** (G_k) is the *optimal* rank-(k) surrogate **under task-relevant criteria**. We will:

1. **Reconfirm** SVD optimality under unitarily invariant norms (Eckart–Young–Mirsky) with precise finite-precision experiments.
2. **Exhibit regimes where SVD is suboptimal**: weighted geometries, robust norms, elementwise trust regions, and tensor structure.
3. **Propose & test geometry-aware alternatives** (e.g., whitened SVD, weighted/projection-space SVD, robust low-rank) that strictly dominate vanilla SVD under those criteria.
4. **Characterize streaming/sequence settings** where maintaining a low-rank subspace across ({G_t}) (Frequent Directions/POD) outperforms per-step SVD for cumulative objectives.

**AM205 fit:** numerical linear algebra + optimization; computing is central; no training of ML models.

---

## 2) Research Questions & Hypotheses

- **RQ1 (Unitarily invariant norms):** Under (|\cdot|_F) and (|\cdot|_2), truncated SVD is optimal.
    
    *H1:* In floating-point with randomized/partial SVD, error matches theory up to rounding; CUR/ID can only match within known bounds.
    
- **RQ2 (Weighted geometry):** For weighted error (|W_L^{1/2}(G-X)W_R^{1/2}|_F) with SPD (W_L,W_R),
    
    *H2:* **Whitened SVD** (X^\star=W_L^{-1/2}, \big[SVD_k(W_L^{1/2}GW_R^{1/2})\big], W_R^{-1/2}) strictly beats vanilla SVD unless (W_L,W_R\propto I). Gap scales with (\kappa(W_L),\kappa(W_R)).
    
- **RQ3 (Descent-oriented criteria):** For one-step linearized loss drop, (\max_{\mathrm{rank}\le k,\ |\Delta|\le \rho} \langle -G,\Delta\rangle):
    
    *H3:* With Frobenius/OP balls, SVD directions are optimal (Ky Fan/von Neumann); with **coordinate-wise trust regions / diagonal preconditioners**, optimal directions come from SVD in the preconditioned space, not vanilla SVD.
    
- **RQ4 (Robustness):** Under sparse outliers/heavy-tailed noise,
    
    *H4:* Robust low-rank (RPCA/IRLS-(L_1)) yields lower (L_1) error and better descent alignment than SVD; provide small constructive counterexamples.
    
- **RQ5 (Tensor structure):** For (G) as 3-/4-D tensors,
    
    *H5:* Matricization+SVD is suboptimal vs HOSVD/CP under multilinear-rank metrics; quantify gaps vs mode condition numbers.
    
- **RQ6 (Streaming):** For a sequence ({G_t}),
    
    *H6:* Frequent Directions/POD maintains a subspace that yields better cumulative alignment and lower memory/time than per-step SVD at equal rank.
    

---

## 3) Methods

### 3.1 Theory (short, AM205-length proofs)

- Re-derive **Eckart–Young–Mirsky** and **Ky Fan** inequalities (sketches).
- Show **weighted low-rank** reduces to SVD via congruence transform; prove strict improvement criteria.
- Provide **von Neumann trace inequality** proof for descent-aligned updates.
- Construct **tight counterexamples** (2×2 or 3×3) where SVD loses under (L_1), bound size of the gap.
- Tensor case: define multilinear rank; show conditions where matricization destroys optimality.

### 3.2 Algorithms to Implement (NumPy/SciPy only)

- Truncated SVD: `svd`, `svds`, randomized SVD (Halko).
- **Whitened SVD** for weighted norms (SPD draws via (Q\Lambda Q^\top)).
- CUR / Interpolative Decomposition (ID) via column/row selection (QR with pivoting + least-squares).
- Robust low-rank: RPCA via inexact ALM or simple IRLS (L_1) scheme.
- Streaming: **Frequent Directions** (compact sketch matrix), plus classic POD on sliding windows.
- Tensor: HOSVD (mode-(k) SVD), CP-ALS (small-scale).

All code pure NumPy/SciPy; optional Tensorly (NumPy backend) for tensor baselines.

---

## 4) Experimental Design

### Synthetic Data Generators (no ML training)

- **Low-rank + noise:** (G = U\Sigma V^\top + \sigma N), controlled spectral decay.
- **Weighted geometry:** sample SPD (W_L,W_R) with target condition numbers.
- **Outliers:** add sparse spikes with controlled magnitude/density.
- **Diagonal trust regions / preconditioning:** elementwise bounds or (P_L,P_R) diagonal SPD.
- **Tensors:** generate rank-((r_1,r_2,r_3)) cores with mode-condition control.
- **Streams:** AR(1) drift in singular vectors/values to simulate evolving gradients.

### Metrics

- **Approximation:** (|G-X|_F), (|G-X|_2); **weighted** (|W_L^{1/2}(G-X)W_R^{1/2}|_F).
- **Descent alignment:** (\cos\angle(G,\Delta)), predicted linearized drop (-\langle G,\Delta\rangle) under constraints.
- **Robustness:** (L_1) error, recovery of clean component, angle to clean (G).
- **Tensor:** multilinear reconstruction error; per-mode principal angles.
- **Compute:** wall-time, flops proxy (via operation counts), memory footprint; stability vs conditioning.

### Core Experiments (rank grid (k\in{1,2,5,10,20}))

A. **SVD-optimal regimes:** SVD vs randomized SVD vs CUR/ID under Fro/OP norms.

B. **Weighted geometry:** vanilla vs whitened SVD; error gap vs (\kappa(W)).

C. **Descent-oriented:** SVD in raw vs preconditioned space under different trust-region shapes.

D. **Robustness:** SVD vs RPCA/IRLS over outlier magnitude/density sweeps.

E. **Tensor:** matricize+SVD vs HOSVD/CP; effect of mode conditioning.

F. **Streaming:** per-step SVD vs Frequent Directions/POD—cumulative alignment & runtime.

---

## 5) Expected Results & Success Criteria

- Confirm **SVD’s dominance** under Fro/OP norms; randomized SVD ≈ SVD; CUR slightly worse but faster/interpretable.
- Show **strict, quantifiable gains** of **whitened SVD** in weighted norms; gap grows with (\kappa(W)).
- Demonstrate **preconditioned-space SVD** is optimal for descent under diagonal trust regions; vanilla SVD can be materially worse.
- Provide **robust counterexamples** where RPCA/IRLS beats SVD under (L_1)/Huber and preserves descent direction.
- Establish cases where **HOSVD/CP** beat matricization+SVD in tensor metrics.
- Show **Frequent Directions** attains better cumulative objectives than per-step SVD at equal rank/time in drifting streams.

**Workshop-grade novelty trigger:** at least one practically relevant regime (weighted/robust/streaming/tensor) where a **simple alternative** (whitened-SVD, RPCA, FD) **strictly dominates** vanilla SVD with clear theory + plots.

---

## 6) Reproducibility & Artifacts

- Deterministic seeds; double precision; report condition numbers; error bars over ≥20 trials.
- Open-source NumPy/SciPy code; one-click scripts to reproduce all figures.
- Document floating-point pitfalls (rounding when spectra are clustered) and how we handle them.

---

## 7) Risks & Mitigations

- **RPCA runtime:** cap sizes (e.g., (m,n\le 2{,}000)), use IRLS fallback.
- **Tensor algorithms instability:** keep low multilinear ranks; verify with synthetic ground truth.
- **Ill-conditioning:** scale inputs; report sensitivity; use randomized SVD to stabilize.

---

## 8) Deliverables (mapped to AM205 grading)

- **Clarity/correctness of coding:** Clean, unit-tested NumPy/SciPy modules.
- **Correctness of results:** Theorem statements + checks against known identities.
- **Effective figures:**
    - Error vs rank (Fro/OP/weighted)
    - Gap vs (\kappa(W))
    - Descent alignment under different constraints
    - Robustness curves vs outlier level
    - Tensor reconstruction bars
    - Streaming cumulative drop & runtime
- **Concise writing:** ≤10 pages (main); proofs/algorithms in appendix.

---

## 9) Tentative Paper Outline (≤10 pages)

1. **Intro & Motivation** (½ p): SVD optimality is metric-dependent; why gradients/tensors/streams matter.
2. **Preliminaries** (1 p): norms, SVD, Ky Fan/von Neumann; problem setups.
3. **When SVD is Optimal** (1 p): theory recap + finite-precision replication.
4. **When SVD is Not** (3 p): weighted geometry, descent under trust regions, robustness, tensors (theory + key figures).
5. **Streaming Subspace Maintenance** (1 p): FD/POD vs per-step SVD.
6. **Computational Study** (2 p): unified benchmarks & results.
7. **Takeaways & Guidelines** (½ p).
8. **Limitations & Future Work** (½ p).

    Appendices: proofs, algorithm listings, extra plots.

---

## 10) Empirical Highlights (latest run)

- **Weighted geometry:** Whitening drops the anisotropic Frobenius objective by roughly 11% (31.7 → 28.3) at the cost of a modest rise in unweighted error, reinforcing the need to match the metric to the task.【F:experiments/results/summary.md†L11-L15】
- **Descent alignment:** Preconditioning improves weighted cosine alignment and inner-product descent compared with vanilla SVD, while slightly lowering the standard cosine, underscoring geometry trade-offs.【F:experiments/results/summary.md†L17-L26】
- **Robustness:** Quantile-trimmed updates shrink the L₁ reconstruction gap from 424 to 31 and the Frobenius gap from 265 to 24, validating the heavy-tail mitigation hypothesis.【F:experiments/results/summary.md†L28-L34】
- **Tensor structure:** Matricized SVD overfits matrix-rank interference (≈0.04% error to the observed tensor) yet stays 54% further from the clean multilinear signal than HOSVD, illustrating the benefit of tensor-aware constraints.【F:experiments/results/summary.md†L36-L42】
- **Streaming:** Frequent Directions maintains a rolling sketch with ≈10× lower cumulative error than a comparably budgeted periodic SVD, while using only eight heavy recomputations versus per-step refits.【F:experiments/results/summary.md†L44-L49】

## 10) Experiments & Empirical Findings

The `experiments/` package implements the empirical demonstrations promised above. Run the full suite (≈20 seconds on a laptop) via:

```bash
python -m experiments.experiment_suite
```

Results are written to `experiments/results/summary.md` and `summary.npy`. The main takeaways from the default seed are:

- **Classical regime (RQ1):** Truncated and randomized SVD remain numerically indistinguishable under Frobenius/Operator norms, whereas a leverage-score CUR baseline trails by roughly 4× in error.【F:experiments/experiment_suite.py†L37-L62】【F:experiments/results/summary.md†L1-L8】
- **Weighted geometry (RQ2):** Ill-conditioned weights make vanilla SVD misspecify the task metric; whitening cuts the weighted loss by ~11% even as the unweighted Frobenius error rises slightly.【F:experiments/experiment_suite.py†L64-L104】【F:experiments/results/summary.md†L10-L16】
- **Descent constraints (RQ3):** Preconditioning improves weighted cosine alignment and descent energy at the expense of standard cosine similarity, illustrating the geometry trade-off explicitly.【F:experiments/experiment_suite.py†L106-L138】【F:experiments/results/summary.md†L17-L26】
- **Robustness (RQ4):** Quantile trimming before the low-rank fit slashes both L₁ and Frobenius error against the clean matrix and nudges cosine alignment upward.【F:experiments/experiment_suite.py†L140-L170】【F:experiments/results/summary.md†L28-L34】
- **Tensor structure (RQ5):** When matrix-rank interference is injected, matricized SVD overfits the corruption (≈0.04% error to the observed tensor) yet stays far from the clean multilinear component, while HOSVD better preserves the planted factors.【F:experiments/experiment_suite.py†L172-L206】【F:experiments/results/summary.md†L36-L42】
- **Streaming gradients (RQ6):** Frequent Directions beats a compute-matched periodic SVD refit by an order of magnitude in cumulative error while requiring only eight heavy recomputations, with the per-step SVD reported as an oracle lower bound.【F:experiments/experiment_suite.py†L208-L266】【F:experiments/results/summary.md†L44-L49】

Use the saved NumPy file or Markdown summary to drive figure generation in the paper draft.

