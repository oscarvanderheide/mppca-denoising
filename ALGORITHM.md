# How the algorithm works

Three levels of explanation — pick the one that matches your background.

---

## Level 1 — Intuition (no equations)

### The basic idea

MRI data collected multiple times under slightly different conditions (different
echo times, diffusion directions, flip angles, …) is highly redundant.  Nearby
voxels in the brain tend to look similar, and the small set of acquisition
parameters means the data really lives in a low-dimensional space even if you
have many measurements.

Noise, on the other hand, is random and fills all dimensions equally.

The algorithm exploits this gap:

1. **Slide a small 3-D window** (e.g. 5×5×5 voxels) across the volume.
2. Inside each window, stack the measured values at every voxel into a matrix:
   rows = voxels, columns = measurements.
3. **Decompose that matrix** in a way that separates the "organised" (signal)
   directions from the "random" (noise) directions.
4. **Keep only the signal directions**, discard the noise, then reconstruct.
5. Repeat for every window position and average the overlapping results.

Step 3–4 is where the statistics come in.  The key insight (Marchenko–Pastur law)
is that for a purely random matrix the distribution of its squared singular values
has a known shape.  Anything larger than that distribution must be signal;
anything inside it is noise.  No tuning parameter is needed — the threshold is
determined automatically from the data.

### Why multiple measurement dimensions help (tMPPCA)

If your acquisition has structure — for example, a grid of b-values × diffusion
directions — you can do the decomposition separately along each axis instead of
treating all measurements as one flat list.  Each axis can have a different
signal rank, and the combined noise estimate is more precise.  That is what the
"t" (tensor) in tMPPCA stands for.

---

## Level 2 — The math (compact)

### Setup

For each window position, collect a patch matrix $X \in \mathbb{R}^{W \times M}$
(or $\mathbb{C}^{W \times M}$ for complex data), where $W = |\text{window}|$ is
the number of voxels and $M$ is the number of measurements.  The model is

$$
X = S + N,
$$

where $S$ is low-rank signal (rank $P \ll \min(W, M)$) and $N$ is i.i.d.
Gaussian noise with variance $\sigma^2$.

### Noise estimation via Marchenko–Pastur

The squared singular values $\lambda_1 \geq \ldots \geq \lambda_K$ of $X$
(where $K = \min(W, M)$) are split into signal and noise by finding the smallest
$P$ such that

$$
\lambda_P > \hat{\sigma}^2_P \left(\sqrt{W} + \sqrt{M}\right)^2,
$$

where $\hat{\sigma}^2_P$ is the noise-variance estimate implied by the tail sum
of eigenvalues:

$$
\hat{\sigma}^2_P = \frac{\sum_{k=P+1}^{K} \lambda_k}{(W - P)(M - P)}.
$$

The algorithm finds the smallest $P$ for which this self-consistent equation
holds — no threshold needs to be set by hand.

### Reconstruction and optimal shrinkage

The patch is reconstructed as

$$
\hat{X} = \sum_{k=1}^{P} \tilde{s}_k \, u_k v_k^H,
$$

where $u_k$, $v_k$ are singular vectors and $\tilde{s}_k$ are optionally
*shrunk* singular values.  The default shrinkage (Gavish & Donoho 2017) sets

$$
\tilde{s}_k = \sqrt{\max\!\left(s_k^2 - 2(W' + M')\sigma^2 - \frac{(W' - M')^2 \sigma^4}{s_k^2},\; 0\right)},
$$

where $W' = W - P$, $M' = M - P$ are the effective noise-matrix dimensions.
This is the asymptotically optimal estimator under the Frobenius loss.

### SNR gain

Keeping $P$ signal components out of $K$ reduces the noise power by a factor

$$
\text{SNR gain} = \sqrt{\frac{W M}{P(W + M - P)}},
$$

which can be substantial when $P \ll W$ or $P \ll M$.

---

## Level 3 — Full technical description

### Algorithm: tMPPCA (Tensor MP-PCA)

This implementation follows Olesen et al. (MRM 2022), which generalises MPPCA to
data with multiple acquisition dimensions.  The core idea is to apply MP-PCA
along each dimensional *mode* of the patch in sequence — a Tucker decomposition
— rather than treating all acquisition dimensions as a single flat measurement
axis.

#### Patch extraction

For a volume of shape $(*D_1, \ldots, D_S, M_1, \ldots, M_A)$, where the first
$S$ dimensions are spatial and the remaining $A$ dimensions are acquisition / 
measurement axes, a rectangular window of size $W = w_1 \times \ldots \times w_S$
slides over the spatial dimensions.  Each window position yields a patch tensor

$$
\mathcal{X} \in \mathbb{F}^{W \times M_1 \times \cdots \times M_A},
$$

where $\mathbb{F}$ is $\mathbb{R}$ or $\mathbb{C}$ depending on the input.

When there is only one acquisition dimension ($A = 1$), the Tucker decomposition
reduces to a single matrix SVD — identical to plain MPPCA.

#### Mode-$n$ unfolding

Given a tensor of shape $(d_0, d_1, \ldots, d_{L-1})$, the *mode-$n$ unfolding*
(or *$i_n$-flattening*) reshapes it to a matrix of shape
$(d_n, \prod_{j \neq n} d_j)$ by moving mode $n$ to the rows and concatenating
all other modes into the columns.

#### Pass 1 — Combined noise estimation

Apply mode-$n$ unfolding to the original patch $\mathcal{X}$ for each mode
$n = 0, 1, \ldots, L-1$ (mode 0 = spatial $W$, modes 1… = $M_1, M_2, \ldots$).
For each unfolded matrix $X_n$ of shape $(d_n, \prod_{j \neq n} d_j)$:

1. Form the smaller gram matrix: $G_n = X_n X_n^H$ if $d_n \leq \prod_{j \neq n} d_j$,
   else $G_n = X_n^H X_n$.
2. Eigendecompose: $G_n = U \Lambda U^H$, giving squared singular values
   $\lambda_1^{(n)} \geq \ldots \geq \lambda_{K_n}^{(n)}$ (in descending order).
3. Apply the MP estimator to obtain an initial signal rank $P_n$ and
   per-mode noise variance.

Combine all modes into a single shared estimate (MATLAB `combined_noise_estimate`):

$$
\sigma^2 = \frac{\displaystyle\sum_{n} \sum_{k > P_n} \lambda_k^{(n)}}
                {\displaystyle\sum_{n} (d_n - P_n)(\tilde{d}_n - P_n)},
$$

where $\tilde{d}_n = \prod_{j \neq n} d_j$.  Then refine each $P_n$ using this
shared $\sigma^2$:

$$
P_n = \#\!\left\{ k : \lambda_k^{(n)} > \sigma^2 \left(\sqrt{d_n} + \sqrt{\tilde{d}_n}\right)^2 \right\}.
$$

#### Pass 2 — Sequential Tucker truncation (forward sweep)

Starting from $\mathcal{X}$, for each mode $n = 0, 1, \ldots, L-1$:

1. Unfold to $X_n$ of shape $(d_n^{\text{cur}}, \cdot)$, where $d_n^{\text{cur}}$
   may have been reduced by a previous iteration.
2. Compute top-$P_n$ left singular vectors $U_n \in \mathbb{F}^{d_n \times P_n}$
   via gram-matrix eigh.
3. Project: $X_n^{\text{next}} = U_n^H X_n$ of shape $(P_n, \cdot)$.
4. Zero out components beyond each patch's individual $P_n$ (since batch elements
   may have different ranks; we slice to $\max_b P_n^{(b)}$ and zero-mask the extras).
5. Refold back to the multi-mode shape with dimension $n$ reduced from $d_n$ to $P_n$.

After all modes, $\mathcal{X}$ has become the Tucker core
$\mathcal{G} \in \mathbb{F}^{P_0 \times P_1 \times \cdots \times P_{L-1}}$.

#### Optional Frobenius-optimal shrinkage

Before reconstruction, apply the Gavish–Donoho shrinkage formula to the last mode
of the Tucker core.  This corrects for the bias in the singular values of an
"almost low-rank" matrix observed in the presence of Gaussian noise (see Level 2
for the formula).  The shrinkage is applied via the eigenvectors of the gram
matrix of the last-mode unfolding of $\mathcal{G}$.

#### Backward pass — reconstruction

Re-expand the Tucker core by multiplying back each $U_n$ factor in reverse order:

$$
\hat{\mathcal{X}} = \mathcal{G} \times_0 U_0 \times_1 U_1 \cdots \times_{L-1} U_{L-1},
$$

where $\times_n$ denotes the mode-$n$ product.  In code this is implemented as:
for $n = L-1$ down to $0$, unfold the current tensor along mode $n$, multiply
$U_n \cdot X_n^{\text{core}}$, and refold.

#### Overlap accumulation and averaging

Each voxel is covered by multiple overlapping patches.  All reconstructed patch
values are accumulated with a running count and divided at the end:

$$
\hat{x}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{p \in \mathcal{N}(v)} \hat{X}_{p,v},
$$

where $\mathcal{N}(v)$ is the set of patches containing voxel $v$.

#### GPU implementation notes

- The outer loop over patches is batched: all patches in one mini-batch are
  stacked to a (B, W, n_meas) tensor and all mode operations are broadcast over
  the batch dimension.
- Gram-matrix eigh scales as $O(B \cdot K^3)$ for a $(B, K, K)$ batch, far
  cheaper than batched SVD on $(B, W, M)$ when $K = \min(W, M) \ll W$.
- On CUDA, `TORCH_LINALG_EIGH_BACKEND=3` selects cuSOLVER `syevjBatched`,
  which is ~25× faster than the default `syevd` heuristic for small batched
  matrices ([PyTorch PR #175403](https://github.com/pytorch/pytorch/pull/175403)).

---

## What is Tucker decomposition?

A matrix has *rank $P$*: it can be written as a sum of $P$ outer products of
two vectors.  SVD finds the best rank-$P$ approximation.

A *tensor* (multi-dimensional array) generalises this.  Instead of one rank
number, it has a *Tucker rank* — a separate rank for each mode (dimension).
The *Tucker decomposition* writes a tensor as a small *core tensor* $\mathcal{G}$
multiplied by one "factor matrix" per mode:

$$
\mathcal{X} \approx \mathcal{G} \times_0 U_0 \times_1 U_1 \cdots \times_{L-1} U_{L-1}.
$$

Think of $U_n$ as the principal components along dimension $n$, and $\mathcal{G}$
as the compressed representation in the combined principal-component space.

**Example.** A diffusion MRI dataset might have shape
$(X, Y, Z, \text{b-values}, \text{directions})$.  A patch has shape
$(W, N_b, N_d)$.  MPPCA flattens the last two axes to $(W, N_b \cdot N_d)$ and
finds a single matrix rank.  Tucker / tMPPCA keeps them separate, estimating a
rank along b-values and a rank along directions independently, then combines
into a core tensor of shape $(P_W, P_b, P_d)$.  The joint noise estimate is
more accurate, and the reconstruction can be tighter.

**HOSVD vs Tucker vs tMPPCA.**
*HOSVD* (Higher-Order SVD) is a specific Tucker decomposition where each $U_n$
is computed by SVD of the mode-$n$ unfolding of the *original* tensor — it is
a one-pass, non-iterative method.
*Tucker ALS* (Alternating Least Squares) iterates to convergence.
*tMPPCA* uses a sequential single-pass approach analogous to HOSVD but with
MP-based rank selection at each step: in each mode the current (partially
compressed) tensor is unfolded, SVD'd, and passed forward.  This is more
computationally efficient than ALS and produces a near-optimal result for
the low-rank-plus-noise matrix model.
