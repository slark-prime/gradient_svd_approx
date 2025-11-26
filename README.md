# When is Truncated SVD the Best Low-Rank Update?
## 1. Motivation and Problem Statement

In many numerical optimization problems, we work with **matrix-shaped gradients or Jacobians**
(G \in \mathbb{R}^{m\times n}) and want to replace them by a **rank-(k)** surrogate for reasons like:

* reducing storage or communication (e.g., only storing / transmitting a low-rank update),
* accelerating a solver by operating in a low-dimensional subspace.

The textbook choice is the **truncated SVD** (G_k), which solves
[
\min_{\operatorname{rank}(X)\le k} |G - X|_F
]
by the Eckart–Young theorem. This is optimal in the **standard Frobenius norm**, i.e. when all entries and directions are treated equally (isotropic geometry).

However, in actual optimization algorithms the “size” of a step or the error is often measured in a **weighted norm** induced by a symmetric positive definite (SPD) matrix:

* **preconditioned gradient methods**: trust region (|W^{1/2}\Delta|_F \le \rho),
* **weighted least squares**: some rows/columns of (G) are more important,
* **diagonal scaling**: each coordinate scaled differently.

Mathematically, this leads to a **weighted Frobenius norm** for matrices:
[
|G - X|_{W_L,W_R}
= \left|,W_L^{1/2}(G-X)W_R^{1/2},\right|_F,
]
with (W_L\succ 0), (W_R\succ 0). In this geometry, the “best” rank-(k) approximation need **not** be the truncated SVD of (G).

By the standard change-of-variables trick, if we define
[
\tilde G = W_L^{1/2} G W_R^{1/2},
]
then
[
\min_{\operatorname{rank}(X)\le k} |W_L^{1/2}(G-X)W_R^{1/2}|*F
\quad\Longleftrightarrow\quad
\min*{\operatorname{rank}(\tilde X)\le k} |\tilde G - \tilde X|_F.
]
The optimizer in the unweighted Frobenius norm is the truncated SVD of (\tilde G):
[
\tilde X^\star = (\tilde G)_k,
]
and mapping back gives
[
X^\star = W_L^{-1/2} \tilde X^\star W_R^{-1/2}.
]
We call this the **whitened SVD solution**: it is just the SVD in a “whitened” (metric-adjusted) space.

### Core questions

1. **Approximation question:**
   Under the weighted Frobenius norm (|\cdot|_{W_L,W_R}), how much better is the **whitened SVD** (X^\star) than the **vanilla truncated SVD** (G_k)? How does the gap depend on the **condition number** of (W_L, W_R)?

2. **Descent / optimization question:**
   If we interpret a rank-(k) matrix (\Delta) as a **step direction** and constrain its size in a weighted norm (a preconditioned trust region), does the **whitened SVD direction** give strictly larger one-step decrease in the linearized loss than a vanilla SVD direction?

3. **Toy counterexample question:**
   Can we show a tiny (2\times 2) or (3\times 3) example where the usual SVD is *not* optimal in the weighted norm, and the whitened SVD is strictly better?

The experiments below are designed to answer exactly these, in a way that’s small, clear, and reproducible.

---

## 2. Overview of Experiments

We’ll run three main experiments:

1. **Experiment 1 (Approximation):**
   Compare vanilla vs whitened SVD under a weighted Frobenius norm and show the gap grows with the conditioning of (W_L, W_R).

2. **Experiment 2 (Descent under trust region):**
   Under a weighted trust-region constraint, compare the **linearized loss drop** from a vanilla-SVD-based step vs a whitened-SVD-based step.

3. **Experiment 3 (Tiny counterexample):**
   Construct an explicit small matrix + SPD weight where whitened SVD beats vanilla SVD in the weighted norm; put values in a table.

All experiments are on **synthetic matrices**; no ML training involved.

---

## 3. Experiment 1 — Weighted Low-Rank Approximation

### Goal

Show empirically that under the weighted Frobenius norm (|\cdot|_{W_L,W_R}),

* the **whitened SVD solution** (X^\star) achieves **lower weighted error** than the vanilla truncated SVD (G_k),
* the **advantage increases** as the SPD weights become more ill-conditioned (anisotropic).

### Setup

* Matrix size: (m = n = 100) (could also test 50 and 200 to see robustness).
* True underlying rank: (r_{\text{true}} = 10).
* Approximation ranks: (k \in {1,2,5,10}).
* Condition numbers for SPD weights: (\kappa \in {1, 5, 10, 50, 100}).
* Number of random trials per configuration: e.g. 20.
* **Two-sided weighting by default:** the code samples **independent SPD matrices** (W_L \in \mathbb{R}^{m\times m}) and (W_R \in \mathbb{R}^{n\times n}) with the same target condition number. This explicitly exercises anisotropy in both the row and column spaces; set `two_sided=False` to fall back to (W_L = W_R).

### Data generation per trial

1. **Generate a low-rank + noise matrix (G):**

   * Draw (U \in \mathbb{R}^{m\times r_{\text{true}}}, V \in \mathbb{R}^{n\times r_{\text{true}}}) with i.i.d. (N(0,1)); orthonormalize via QR to get orthonormal columns.
   * Define singular values with geometric decay:
     [
     \sigma_i = \alpha^{i-1}, \quad \alpha \in [0.5, 0.9].
     ]
   * Set (G_0 = U ,\text{diag}(\sigma_1,\dots,\sigma_{r_{\text{true}}}) V^\top).
   * Add small noise: (G = G_0 + \sigma_{\text{noise}} N), where (N) has i.i.d. (N(0,1)), and e.g. (\sigma_{\text{noise}} = 0.01).

2. **Generate SPD weights (W_L, W_R):**

   * Sample a random orthogonal matrix (Q) via QR on a Gaussian matrix **independently for the left and right metrics** (unless you set `two_sided=False`).
   * Choose eigenvalues (\lambda_i) between 1 and (\kappa), e.g. log-spaced:
     [
     \lambda_i = \exp\left(\log 1 + \frac{i-1}{m-1}\log \kappa\right).
     ]
   * Define (W_L = Q_L ,\text{diag}(\lambda_i) Q_L^\top) and (W_R = Q_R ,\text{diag}(\lambda_i) Q_R^\top). Two-sided metrics highlight when whitening genuinely needs to happen on both sides.

### Methods to compare

For each (G, W_L, W_R, k):

1. **Vanilla truncated SVD:**

   * Compute SVD of (G):
     [
     G = U \Sigma V^\top.
     ]
   * Truncate to rank (k):
     [
     G_k = U_k \Sigma_k V_k^\top.
     ]

2. **Whitened SVD (the weighted-optimal solution):**

   * Compute SPD square roots (W_L^{1/2}, W_R^{1/2}) and their inverses (via eigen-decomposition or Cholesky).
   * Form the whitened matrix:
     [
     \tilde G = W_L^{1/2} G W_R^{1/2}.
     ]
   * Compute SVD:
     [
     \tilde G = \tilde U \tilde \Sigma \tilde V^\top.
     ]
   * Truncate:
     [
     \tilde G_k = \tilde U_k \tilde \Sigma_k \tilde V_k^\top.
     ]
   * Map back to the original space:
     [
     X^\star = W_L^{-1/2} \tilde G_k W_R^{-1/2}.
     ]

### Metrics

For each method (vanilla, whitened):

* **Weighted reconstruction error:**
  [
  E_{\text{vanilla}} = |W_L^{1/2}(G - G_k)W_R^{1/2}|*F,
  \quad
  E*{\text{white}} = |W_L^{1/2}(G - X^\star)W_R^{1/2}|_F.
  ]

Optionally:

* **Unweighted Frobenius error** (|G - G_k|_F) vs (|G - X^\star|_F) to illustrate that:

  * SVD is optimal in the unweighted norm,
  * whitened SVD is optimal in the weighted norm.

Aggregate:

* For each ((k, \kappa)), compute the **average** and **standard deviation** of (E_{\text{vanilla}}, E_{\text{white}}) over all trials.

### Plots

1. **Weighted error vs rank (fixed (\kappa)):**

   * x-axis: rank (k),
   * y-axis: mean weighted error,
   * curves: vanilla SVD vs whitened SVD.

2. **Error gap vs condition number (fixed (k)):**

   * x-axis: (\kappa(W)),
   * y-axis: mean difference (E_{\text{vanilla}} - E_{\text{white}}),
   * include error bars if you like.

**Expected qualitative result:**

* For (\kappa=1) (isotropic case), the two methods coincide (or are very close numerically).
* As (\kappa) grows, (E_{\text{white}}) stays minimal (by theory), and (E_{\text{vanilla}}) gets noticeably worse in the weighted sense.

---

## 4. Experiment 2 — Descent Under a Weighted Trust Region

### Goal

Show that if you think of low-rank approximations as **updates** (\Delta) in an optimization step, then under a **weighted trust-region constraint**, the whitened-SVD-based update achieves **larger one-step linearized loss decrease** than a vanilla-SVD-based update.

### Setup

We consider a one-step model:

* We want a rank-(k) update (\Delta) that maximizes the linearized loss drop:
  [
  \max_{\operatorname{rank}(\Delta)\le k,\ |W^{1/2}\Delta|_F \le \rho}
  -\langle G,\Delta\rangle.
  ]
* (W\succ 0) is an SPD matrix defining the geometry; (\rho>0) is a radius.

We’ll compare two candidate updates:

* (\Delta_{\text{vanilla}}): built from the vanilla truncated SVD of (G).
* (\Delta_{\text{white}}): built from the truncated SVD of the whitened gradient (\tilde G = W^{1/2}G), then mapped back.

### Data generation

Reuse the same scheme as Experiment 1:

* Generate (G) as low-rank + noise.
* Generate SPD weight (W) with condition number (\kappa\in{1,5,10,50,100}).
* Choose a fixed trust-region radius (\rho) (e.g. (\rho = 1)); the exact value doesn’t matter because we’ll scale directions to meet the constraint.

### Constructing the updates

1. **Vanilla SVD-based update (\Delta_{\text{vanilla}})**

* Compute truncated SVD of (G): (G_k = U_k \Sigma_k V_k^\top).
* Direction (before scaling): (D_{\text{vanilla}} = -G_k).
* Scale so that the weighted norm meets the constraint:
  [
  \alpha = \frac{\rho}{|W^{1/2} D_{\text{vanilla}}|*F},
  \quad
  \Delta*{\text{vanilla}} = \alpha D_{\text{vanilla}}.
  ]

2. **Whitened-SVD-based update (\Delta_{\text{white}})**

* Whiten the gradient: (\tilde G = W^{1/2} G).
* Compute truncated SVD: (\tilde G_k = \tilde U_k \tilde\Sigma_k \tilde V_k^\top).
* In whitened coordinates, the best direction under (|\cdot|_F \le \rho) is proportional to (-\tilde G_k):
  [
  \tilde D^\star = -\tilde G_k.
  ]
* Scale in whitened space:
  [
  \beta = \frac{\rho}{|\tilde D^\star|_F} = \frac{\rho}{|\tilde G_k|_F},
  \quad
  \tilde \Delta^\star = \beta \tilde D^\star.
  ]
* Map back:
  [
  \Delta_{\text{white}} = W^{-1/2} \tilde \Delta^\star.
  ]
  Both (\Delta_{\text{vanilla}}) and (\Delta_{\text{white}}) satisfy (|W^{1/2}\Delta|_F = \rho) by construction.

### Metrics

For each trial:

* **Linearized loss drop:**
  [
  D_{\text{vanilla}} = -\langle G, \Delta_{\text{vanilla}}\rangle,
  \quad
  D_{\text{white}} = -\langle G, \Delta_{\text{white}}\rangle.
  ]
* Optionally, **alignment**:
  [
  \cos\theta_{\bullet} = \frac{\langle G,\Delta_{\bullet}\rangle}
  {|G|*F ,|\Delta*{\bullet}|_F}.
  ]

Aggregate over trials for each ((k,\kappa)).

### Plots

* For each fixed (k), plot:

  * x-axis: (\kappa(W)),
  * y-axis: mean (D_\bullet),
  * curves: (D_{\text{vanilla}}) vs (D_{\text{white}}).

**Expected result:**

* When (\kappa=1), they should be equal (or very close).
* As (\kappa) increases, (D_{\text{white}}) consistently exceeds (D_{\text{vanilla}}), showing that whitened-SVD-based directions are better steepest-descent steps in the weighted trust-region sense.

---

## 5. Experiment 3 — Tiny Counterexample (2×2 or 3×3)

### Goal

Provide a **small, exact example** showing that:

* Under the weighted norm, vanilla truncated SVD is *not* optimal,
* The whitened SVD solution achieves strictly smaller weighted error.

### Design

Pick something like:

* (W = \begin{pmatrix} 1 & 0 \ 0 & \kappa \end{pmatrix}) with (\kappa) large (e.g. 100).
* A simple (2\times 2) matrix (G), e.g.
  [
  G = \begin{pmatrix}
  1 & a \
  b & c
  \end{pmatrix}
  ]
  with values chosen so that the top singular vector is not aligned with the direction that matters under the weighted norm (you can pick something and adjust until the effect is visible).

Steps:

1. Compute the rank-1 truncated SVD (G_1).
2. Compute the whitened matrix (\tilde G = W^{1/2} G), its rank-1 truncated SVD (\tilde G_1), and the mapped-back solution (X^\star = W^{-1/2} \tilde G_1).
3. Compute and compare:
   [
   |W^{1/2}(G-G_1)|_F, \quad |W^{1/2}(G-X^\star)|_F.
   ]

### Presentation

Include a small table in your report:

| Method       | Approximation matrix (X) | Weighted error (|W^{1/2}(G-X)|_F) |
| ------------ | ------------------------ | --------------------------------- |
| Vanilla SVD  | (numbers)                | (value)                           |
| Whitened SVD | (numbers)                | (smaller value)                   |

This makes the whole story extremely concrete: even in 2×2 you can see SVD is not “best” once the metric is anisotropic.

---

That’s the full, self-contained package:

* Clear **motivation** (weighted norms from preconditioning),
* A clean **problem statement** (when is truncated SVD optimal vs whitened SVD),
* And **experiment designs** that directly match your theoretical claims, easy to code, and easy for someone else to understand and reproduce.


---

## Running the provided experiments

The repository now ships a single driver (`experiments.py`) that implements all three experiments in this document using only
NumPy and Matplotlib. The emphasis is on reproducible scientific-computing workflows (QR-based orthonormal factors, explicit
SPD eigen decompositions, deterministic seeds) rather than any ML-specific tooling.

### Quick start

```bash
pip install -r requirements.txt
python experiments.py --output results
```

This will create `results/exp1`, `results/exp2`, and `results/exp3` containing CSV summaries and publication-style plots for the
approximation and trust-region studies, plus a text table for the 2×2 counterexample.

### Tuning choices

* **Geometric SPD spectra:** We sample eigenvalues between 1 and κ on a log scale to control anisotropy smoothly and keep the
  matrix conditioning explicit in the plots.
* **Low-rank signal + small noise:** A geometric singular-value decay (α=0.7) with tiny Gaussian noise follows the toy design in
  the write-up while keeping the SVDs numerically stable across 20 trials per configuration.
* **Trust-region normalization:** Steps are rescaled so that both vanilla and whitened updates sit on the same weighted
  constraint boundary, isolating the effect of the metric on the linearized loss drop.

Adjust `ExperimentConfig` inside `experiments.py` to explore different sizes, trial counts, or noise levels; every output file is
labelled with κ and rank so new sweeps remain easy to analyze.
