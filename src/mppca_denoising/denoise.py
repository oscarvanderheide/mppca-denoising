"""MP-PCA denoising of MRI data.

Implements the sliding-window approach from:
    Olesen et al., "Tensor denoising of multidimensional MRI data",
    Magn Reson Med, 2022. doi:10.1002/mrm.29478
"""

from __future__ import annotations

import os
from math import prod

import numpy as np
import torch

# Force cuSOLVER syevjBatched for CUDA eigh — ~25x faster than the default syevd
# heuristic for batched small matrices (our (B, K, K) gram matrices, K = min(W, M)).
# https://github.com/pytorch/pytorch/pull/175403
os.environ.setdefault("TORCH_LINALG_EIGH_BACKEND", "3")


@torch.no_grad()
def denoise_tensor(
    data: torch.Tensor | np.ndarray,
    window: list[int] | tuple[int, ...],
    *,
    mask: torch.Tensor | np.ndarray | None = None,
    stride: list[int] | tuple[int, ...] | None = None,
    center_assign: bool = False,
    opt_shrink: bool = True,
    sigma2: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise MRI data of shape (*spatial_dims, M) using MP-PCA.

    Args:
        data:          Shape (*spatial_dims, M). First len(window) dims are spatial.
        window:        Sliding window size per spatial dim.
        mask:          Boolean mask over spatial dims; patches with no masked voxels are skipped.
        stride:        Step size per spatial dim (default: 1).
        center_assign: If True, only accumulate the centre voxel of each patch.
        opt_shrink:    Apply Frobenius-optimal singular-value shrinkage.
        sigma2:        Known noise variance; estimated from data if None.
        device:        PyTorch device.
        dtype:         Computation dtype. Defaults to float32/complex64 on CUDA, else input dtype.
        batch_size:    Patches per GPU kernel call.

    Returns:
        denoised:     Denoised array, same shape as input.
        sigma2_map:   Noise variance per voxel, shape spatial_dims.
        n_signal_map: Signal component count per voxel, shape spatial_dims.
        snr_gain_map: Estimated SNR improvement per voxel, shape spatial_dims.
    """

    # Validate inputs and move to target device and dtype.
    data, window, mask, stride, device, dtype = _prepare_inputs(
        data, window, mask, stride, device, dtype
    )
    n_spatial = len(window)
    spatial_shape = data.shape[:n_spatial]
    n_meas = prod(data.shape[n_spatial:])  # measurements per voxel
    patch_size = prod(window)  # W: voxels per patch
    n_vox = prod(spatial_shape)

    # Flatten spatial dims so every voxel is a row — patch extraction becomes a single index op.
    data_flat = data.reshape(n_vox, n_meas)  # (n_vox, n_meas)
    mask_flat = mask.reshape(n_vox)

    # Precompute the flat index offsets for every voxel inside a patch relative to its
    # corner, and find the centre voxel's position in that offset list.
    patch_offsets, centre_offset = _build_patch_offsets(spatial_shape, window, device)

    # Enumerate all corner positions whose patch fits inside the volume and sits on the
    # stride grid — these are exactly the patches we will process.
    patch_corners = _valid_patch_corners(spatial_shape, window, stride, device)
    n_patches = len(patch_corners)

    # Allocate overlap-accumulation buffers for the sliding-window average.
    real_dtype = torch.zeros(1, dtype=dtype).real.dtype
    denoised_acc = torch.zeros_like(data_flat)
    count_acc = torch.zeros(n_vox, dtype=real_dtype, device=device)
    sigma2_acc = torch.zeros(n_vox, dtype=real_dtype, device=device)
    n_signal_acc = torch.zeros(n_vox, dtype=real_dtype, device=device)

    # Denoise patches in mini-batches and scatter-accumulate the results.
    # Building index tensors per batch bounds peak memory.
    n_done = 0
    for b_start in range(0, n_patches, batch_size):
        corners_b = patch_corners[b_start : b_start + batch_size]
        vox_inds = corners_b.unsqueeze(1) + patch_offsets.unsqueeze(
            0
        )  # (B, patch_size)

        # Skip patches that contain no masked voxels.
        active = (
            mask_flat[vox_inds[:, centre_offset]]
            if center_assign
            else mask_flat[vox_inds].any(1)
        )
        if not active.any():
            n_done += len(corners_b)
            continue
        vox_inds = vox_inds[active]

        patches = data_flat[vox_inds]  # (B, patch_size, n_meas)
        denoised_patches, sigma2_b, n_signal_b = _denoise_patches(
            patches, opt_shrink=opt_shrink, sigma2=sigma2
        )

        if center_assign:
            # Assign only the denoised centre voxel of each patch — no overlap between patches.
            centre_inds = vox_inds[:, centre_offset]
            denoised_acc.scatter_add_(
                0,
                centre_inds.unsqueeze(1).expand(-1, n_meas),
                denoised_patches[:, centre_offset, :],
            )
            count_acc.scatter_add_(
                0,
                centre_inds,
                torch.ones(len(centre_inds), dtype=real_dtype, device=device),
            )
            sigma2_acc.scatter_add_(0, centre_inds, sigma2_b)
            n_signal_acc.scatter_add_(0, centre_inds, n_signal_b.to(real_dtype))
        else:
            # Accumulate every voxel's denoised estimate from all patches it belongs to.
            flat_inds = vox_inds.reshape(-1)
            denoised_acc.scatter_add_(
                0,
                flat_inds.unsqueeze(1).expand(-1, n_meas),
                denoised_patches.reshape(-1, n_meas),
            )
            count_acc.scatter_add_(
                0,
                flat_inds,
                torch.ones(len(flat_inds), dtype=real_dtype, device=device),
            )
            sigma2_acc.scatter_add_(
                0, flat_inds, sigma2_b.unsqueeze(1).expand(-1, patch_size).reshape(-1)
            )
            n_signal_acc.scatter_add_(
                0,
                flat_inds,
                n_signal_b.to(real_dtype)
                .unsqueeze(1)
                .expand(-1, patch_size)
                .reshape(-1),
            )

        n_done += len(corners_b)
        print(
            f"\r  Patches: {n_done}/{n_patches} ({100 * n_done / n_patches:.0f}%)",
            end="",
            flush=True,
        )

    print(f"\r  Patches: {n_patches}/{n_patches} (100%)")

    # Divide by overlap count to get each voxel's average denoised estimate.
    # Voxels touched by no patch (boundary region) keep their original value.
    unvisited = count_acc == 0
    denoised_acc[unvisited] = data_flat[unvisited]
    sigma2_acc[unvisited] = float("nan")
    n_signal_acc[unvisited] = float("nan")
    count_acc[unvisited] = 1
    denoised_acc /= count_acc.unsqueeze(1)
    sigma2_acc /= count_acc
    n_signal_acc /= count_acc

    # Theoretical SNR gain: sqrt(W*M / (P*(W + M - P)))  (Olesen et al., Eq. 6).
    snr_gain = torch.sqrt(
        torch.tensor(float(patch_size * n_meas), dtype=real_dtype, device=device)
        / (n_signal_acc * (patch_size + n_meas - n_signal_acc))
    )

    return (
        denoised_acc.reshape(data.shape),
        sigma2_acc.reshape(spatial_shape),
        n_signal_acc.reshape(spatial_shape),
        snr_gain.reshape(spatial_shape),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _is_cuda(device: torch.device | str | None) -> bool:
    """Return True if device refers to a CUDA device."""
    if isinstance(device, str):
        return "cuda" in device
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return False


def _build_patch_offsets(
    spatial_shape: tuple[int, ...],
    window: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Flat-index offsets for every voxel inside a patch, plus the centre voxel's position.

    For a patch rooted at corner c, ``c + offsets`` gives the flat indices of all
    prod(window) voxels in that patch.
    """
    vol_strides = _c_strides(spatial_shape, device)
    win_strides = _c_strides(tuple(window), device)
    patch_size = prod(window)

    voxel_subs = _ind2sub_c(
        window, torch.arange(patch_size, device=device)
    )  # (patch_size, n_spatial)
    offsets = (voxel_subs * vol_strides).sum(1)  # (patch_size,)
    centre_subs = torch.tensor([(w - 1) // 2 for w in window], device=device)
    centre_offset = int((centre_subs * win_strides).sum().item())
    return offsets, centre_offset


def _valid_patch_corners(
    spatial_shape: tuple[int, ...],
    window: list[int],
    stride: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Flat indices of all patch corner voxels whose patch fits in the volume and lies on the stride grid."""
    n_vox = prod(spatial_shape)
    all_subs = _ind2sub_c(
        spatial_shape, torch.arange(n_vox, device=device)
    )  # (n_vox, n_spatial)
    win_t = torch.tensor(window, device=device)
    dim_t = torch.tensor(list(spatial_shape), device=device)
    str_t = torch.tensor(stride, device=device)
    in_bounds = ((all_subs + win_t) <= dim_t).all(1)
    on_grid = (all_subs % str_t == 0).all(1)
    return torch.where(in_bounds & on_grid)[0]


# ---------------------------------------------------------------------------
# Batched patch denoising
# ---------------------------------------------------------------------------


def _denoise_patches(
    patches: torch.Tensor,
    *,
    opt_shrink: bool,
    sigma2: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise a batch of patches via gram-matrix eigendecomposition.

    Uses eigh on the (B, K, K) gram matrix (K = min(W, M)) instead of SVD on
    (B, W, M).  For W >> M (typical MRI: 125 voxels × 5 measurements), K = M = 5
    so the eigenproblem is tiny; the heavy work stays in cuBLAS matmuls.

    Args:
        patches:    (B, W, M) — B patches, W voxels each, M measurements.
        opt_shrink: Apply Frobenius-optimal singular-value shrinkage.
        sigma2:     Known noise variance, or None to estimate via the MP distribution.

    Returns:
        denoised_patches: (B, W, M).
        sigma2_b:         (B,) noise variance per patch.
        n_signal_b:       (B,) signal component count per patch.
    """
    B, W, M = patches.shape
    device = patches.device
    real_dtype = patches.real.dtype if patches.is_complex() else patches.dtype
    K = min(W, M)

    if K == 1:
        return (
            patches.clone(),
            torch.zeros(B, dtype=real_dtype, device=device),
            torch.ones(B, dtype=torch.long, device=device),
        )

    # Form the (K×K) gram matrix on the smaller dimension.
    # eigh returns eigenvalues in ascending order; we flip to descending.
    # mH is the conjugate transpose (= transpose for real inputs).
    if W >= M:
        gram = patches.mH @ patches  # (B, M, M)
        sq_singvals, V = torch.linalg.eigh(gram)
        sq_singvals = sq_singvals.flip(-1).clamp(min=0)  # (B, K) descending
        V = V.flip(-1)  # (B, M, K) right singular vectors
        XV = patches @ V  # (B, W, K): col k ≈ U_k * s_k
    else:
        gram = patches @ patches.mH  # (B, W, W)
        sq_singvals, U = torch.linalg.eigh(gram)
        sq_singvals = sq_singvals.flip(-1).clamp(min=0)  # (B, K) descending
        U = U.flip(-1)  # (B, W, K) left singular vectors
        XV = U.mH @ patches  # (B, K, M): row k ≈ s_k * V_kᴴ

    # Estimate the number of signal components and noise variance via the Marchenko-Pastur law.
    if sigma2 is None:
        n_signal_b, sigma2_b = _mp_estimate(sq_singvals, W, M)
    else:
        sigma2_b = torch.full((B,), sigma2, dtype=sq_singvals.dtype, device=device)
        n_signal_b = (
            (sq_singvals > sigma2 * (W**0.5 + M**0.5) ** 2).sum(1).to(torch.long)
        )

    # Boolean mask: True for signal components (k < n_signal_b), False for noise.
    signal_mask = torch.arange(K, device=device).unsqueeze(0) < n_signal_b.unsqueeze(
        1
    )  # (B, K)

    # Compute shrunk singular values: signal components are shrunk, noise ones are zeroed.
    if opt_shrink:
        shrunk_singvals = _opt_shrink_batched(
            sq_singvals, n_signal_b, sigma2_b, W, M, signal_mask
        )
    else:
        shrunk_singvals = sq_singvals.sqrt() * signal_mask.to(sq_singvals.dtype)

    # Per-component amplitude scale: shrunk_s_k / s_k  (zero-safe divide).
    singvals = sq_singvals.sqrt()
    amplitude_scale = shrunk_singvals / singvals.masked_fill(
        singvals == 0, 1.0
    )  # (B, K)

    # Reconstruct the denoised patch from the scaled components.
    if W >= M:
        denoised_patches = (XV * amplitude_scale.unsqueeze(1)) @ V.mH  # (B, W, M)
    else:
        denoised_patches = U @ (XV * amplitude_scale.unsqueeze(-1))  # (B, W, M)

    return denoised_patches, sigma2_b, n_signal_b


def _mp_estimate(
    sq_singvals: torch.Tensor, W: int, M: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate signal component count and noise variance via the Marchenko-Pastur law.

    For each candidate split P, the implied noise variance is derived from the tail
    eigenvalue sum.  The smallest P whose P-th eigenvalue still exceeds the corresponding
    MP upper edge is returned as the signal rank.

    Args:
        sq_singvals: (B, K) squared singular values, descending.
        W:           Patch rows (voxels per patch).
        M:           Patch columns (measurements).

    Returns:
        n_signal: (B,) long  — signal component count per patch.
        sigma2_b: (B,) float — noise variance estimate per patch.
    """
    B, K = sq_singvals.shape
    device, dtype = sq_singvals.device, sq_singvals.dtype

    # Tail energy: sum of sq_singvals[k:] for each candidate split index k.
    tail_energy = sq_singvals.flip(1).cumsum(1).flip(1)  # (B, K)

    # Implied noise variance for each candidate P: σ² = Σλ[P:] / ((W-P)*(M-P)).
    p_range = torch.arange(K, device=device, dtype=dtype)
    noise_denom = ((W - p_range) * (M - p_range)).clamp(min=1)
    sigma2_cands = tail_energy / noise_denom  # (B, K)

    # MP upper edge: λ ≤ σ² * (√W + √M)².
    mp_cutoff = sigma2_cands * (W**0.5 + M**0.5) ** 2  # (B, K)

    # The first eigenvalue that falls below the cutoff marks where the noise floor starts.
    below_cutoff = sq_singvals < mp_cutoff  # (B, K)
    any_below = below_cutoff.any(1)  # (B,)
    n_signal = torch.where(
        any_below,
        below_cutoff.long().argmax(1),
        torch.full((B,), K, device=device, dtype=torch.long),
    )

    # Read off the noise variance at the estimated rank (clamp index to stay in-bounds).
    idx = n_signal.clamp(max=K - 1)
    sigma2_b = sigma2_cands[torch.arange(B, device=device), idx]
    sigma2_b = torch.where(
        any_below, sigma2_b, torch.zeros(B, dtype=dtype, device=device)
    )
    return n_signal, sigma2_b


def _opt_shrink_batched(
    sq_singvals: torch.Tensor,
    n_signal_b: torch.Tensor,
    sigma2_b: torch.Tensor,
    W: int,
    M: int,
    signal_mask: torch.Tensor,
) -> torch.Tensor:
    """Frobenius-optimal singular value shrinkage (Gavish & Donoho, 2017), batched.

    Signal components are shrunk towards zero; noise components are zeroed.

    Args:
        sq_singvals: (B, K) squared singular values.
        n_signal_b:  (B,) signal component count per patch.
        sigma2_b:    (B,) noise variance per patch.
        W, M:        Patch dimensions (rows, columns).
        signal_mask: (B, K) bool — True for signal components.

    Returns:
        shrunk_singvals: (B, K) shrunk singular values.
    """
    # Effective noise matrix size after removing signal components.
    n_noise_rows = (
        (W - n_signal_b).clamp(min=1).to(sq_singvals.dtype).unsqueeze(1)
    )  # (B, 1)
    n_noise_cols = (
        (M - n_signal_b).clamp(min=1).to(sq_singvals.dtype).unsqueeze(1)
    )  # (B, 1)
    noise_var = sigma2_b.unsqueeze(1)  # (B, 1)

    # Replace noise-component entries with 1 to avoid division by zero in the formula.
    vals_safe = sq_singvals.clone()
    vals_safe[~signal_mask] = 1.0

    shrunk_sq = (
        vals_safe
        - 2 * (n_noise_rows + n_noise_cols) * noise_var
        + (n_noise_rows - n_noise_cols) ** 2 * noise_var**2 / vals_safe
    )

    # Clamp negatives (sub-noise eigenvalues) and zero out noise components.
    shrunk_singvals = shrunk_sq.clamp(min=0).sqrt() * signal_mask.to(sq_singvals.dtype)
    return shrunk_singvals


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def _prepare_inputs(data, window, mask, stride, device, dtype=None):
    """Convert inputs to tensors on the target device and dtype."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())
    if device is None:
        device = data.device if data.device.type != "cpu" else torch.device("cpu")
    if isinstance(device, str):
        device = torch.device(device)
    if dtype is None:
        dtype = (
            (torch.complex64 if data.is_complex() else torch.float32)
            if device.type == "cuda"
            else data.dtype
        )
    data = data.to(dtype=dtype, device=device)

    window = list(window)
    n_spatial = len(window)
    spatial_shape = tuple(data.shape[:n_spatial])

    if mask is None:
        mask = torch.ones(spatial_shape, dtype=torch.bool, device=device)
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask.copy()).to(dtype=torch.bool, device=device)
    else:
        mask = mask.to(dtype=torch.bool, device=device)

    stride = [1] * n_spatial if stride is None else list(stride)
    return data, window, mask, stride, device, dtype


def _c_strides(
    shape: tuple[int, ...] | list[int], device: torch.device
) -> torch.Tensor:
    """C-order (row-major) strides for a tensor of `shape`, as a 1-D long tensor."""
    ndim = len(shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return torch.tensor(strides, dtype=torch.long, device=device)


def _ind2sub_c(shape: list[int] | tuple[int, ...], idx: torch.Tensor) -> torch.Tensor:
    """C-order linear indices → subscripts, 0-based. Returns (N, ndim) long tensor."""
    ndim = len(shape)
    subs = torch.zeros(len(idx), ndim, dtype=torch.long, device=idx.device)
    remainder = idx.clone()
    for d in range(ndim - 1, -1, -1):
        subs[:, d] = remainder % shape[d]
        remainder = remainder // shape[d]
    return subs
