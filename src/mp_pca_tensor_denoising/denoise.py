"""MP-PCA denoising of MRI data.

Implements the sliding-window approach from:
    Olesen et al., "Tensor denoising of multidimensional MRI data",
    Magn Reson Med, 2022. doi:10.1002/mrm.29478

Usage is free but please cite the paper above.

Assumes data shape (*spatial_dims, M): any number of spatial dims followed by a
single measurement dimension. All patches are denoised in parallel via batched SVD.
"""

from __future__ import annotations

from math import prod

import numpy as np
import torch


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
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise MRI data of shape (*spatial_dims, M) using MP-PCA.

    Args:
        data: Shape (*spatial_dims, M). First ``len(window)`` dims are spatial.
        window: Sliding window size over the spatial dims.
        mask: Boolean mask over the spatial dims. Patches with no True voxels are skipped.
        stride: Step size per spatial dim (default: 1).
        center_assign: If True, only accumulate the denoised centre voxel of each patch.
        opt_shrink: Apply optimal Frobenius-norm singular-value shrinkage.
        sigma2: Known noise variance; estimated from data if None.
        device: PyTorch device. Defaults to the data's device or CPU.
        batch_size: Number of patches processed per kernel call.

    Returns:
        denoised: Same shape as input.
        Sigma2: Noise variance per voxel, shape ``spatial_dims``.
        P: Signal component count per voxel, shape ``spatial_dims``.
        SNR_gain: Estimated SNR improvement per voxel, shape ``spatial_dims``.
    """
    data, window, mask, stride, device, dtype = _prepare_inputs(
        data, window, mask, stride, device
    )
    dims = data.shape
    n_spatial = len(window)
    dims_vox = dims[:n_spatial]
    M_meas = prod(dims[n_spatial:])
    W = prod(window)
    num_vox = prod(dims_vox)

    data_2d = data.reshape(num_vox, M_meas)
    mask_flat = mask.reshape(num_vox)

    # C-order strides for the spatial volume and for the window
    sp_strides = _c_strides(dims_vox, device)
    win_strides = _c_strides(tuple(window), device)

    # Linear offsets of every voxel within a (window-shaped) patch
    window_subs = _ind2sub_c(window, torch.arange(W, device=device))  # (W, n_spatial)
    index_increments = (window_subs * sp_strides).sum(1)  # (W,)

    # Centre voxel: C-order position within the W-element window array
    centre_subs = torch.tensor([(w - 1) // 2 for w in window], device=device)
    centre_ind = int((centre_subs * win_strides).sum().item())

    # --- Vectorised validity filter over all voxel positions ---
    all_subs = _ind2sub_c(
        dims_vox, torch.arange(num_vox, device=device)
    )  # (num_vox, n_spatial)
    win_t = torch.tensor(window, device=device)
    dim_t = torch.tensor(list(dims_vox), device=device)
    str_t = torch.tensor(stride, device=device)
    geom_valid = ((all_subs + win_t) <= dim_t).all(1) & (all_subs % str_t == 0).all(1)
    valid_corners = torch.where(geom_valid)[0]  # (N_corners,)
    N_corners = len(valid_corners)

    # --- Accumulators ---
    denoised = torch.zeros_like(data_2d)
    count = torch.zeros(num_vox, dtype=dtype, device=device)
    Sigma2 = torch.zeros(num_vox, dtype=dtype, device=device)
    P_out = torch.zeros(num_vox, dtype=dtype, device=device)

    # --- Mini-batch loop ---
    # Patch index matrices are built on-the-fly per batch to bound peak memory,
    # then filtered by the mask before extraction.
    n_done = 0
    for b_start in range(0, N_corners, batch_size):
        corners_b = valid_corners[b_start : b_start + batch_size]
        vox_inds_b = corners_b.unsqueeze(1) + index_increments.unsqueeze(0)  # (B, W)

        # Mask filter
        if center_assign:
            mask_ok = mask_flat[vox_inds_b[:, centre_ind]]
        else:
            mask_ok = mask_flat[vox_inds_b].any(1)
        vox_inds_b = vox_inds_b[mask_ok]
        if vox_inds_b.numel() == 0:
            n_done += len(corners_b)
            continue

        patches = data_2d[vox_inds_b]  # (B', W, M_meas)
        patches_den, s2_b, p_b = _denoise_patches(
            patches, opt_shrink=opt_shrink, sigma2=sigma2
        )

        if center_assign:
            c_inds = vox_inds_b[:, centre_ind]  # (B',)
            denoised.scatter_add_(
                0, c_inds.unsqueeze(1).expand(-1, M_meas), patches_den[:, centre_ind, :]
            )
            count.scatter_add_(
                0, c_inds, torch.ones(len(c_inds), dtype=dtype, device=device)
            )
            Sigma2.scatter_add_(0, c_inds, s2_b)
            P_out.scatter_add_(0, c_inds, p_b.to(dtype))
        else:
            flat_inds = vox_inds_b.reshape(-1)  # (B'*W,)
            denoised.scatter_add_(
                0,
                flat_inds.unsqueeze(1).expand(-1, M_meas),
                patches_den.reshape(-1, M_meas),
            )
            count.scatter_add_(
                0, flat_inds, torch.ones(len(flat_inds), dtype=dtype, device=device)
            )
            Sigma2.scatter_add_(
                0, flat_inds, s2_b.unsqueeze(1).expand(-1, W).reshape(-1)
            )
            P_out.scatter_add_(
                0, flat_inds, p_b.to(dtype).unsqueeze(1).expand(-1, W).reshape(-1)
            )

        n_done += len(corners_b)
        print(
            f"\r  Patches: {n_done}/{N_corners} ({100 * n_done / N_corners:.0f}%)",
            end="",
            flush=True,
        )

    print(f"\r  Patches: {N_corners}/{N_corners} (100%)")

    # --- Average overlapping contributions ---
    skipped = count == 0
    denoised[skipped] = data_2d[skipped]
    Sigma2[skipped] = float("nan")
    P_out[skipped] = float("nan")
    count[skipped] = 1
    denoised /= count.unsqueeze(1)
    Sigma2 /= count
    P_out /= count

    # SNR gain: sqrt(W*M / (P*(W + M - P)))
    SNR_gain = torch.sqrt(
        torch.tensor(float(W * M_meas), dtype=dtype, device=device)
        / (P_out * (W + M_meas - P_out))
    )

    return (
        denoised.reshape(dims),
        Sigma2.reshape(dims_vox),
        P_out.reshape(dims_vox),
        SNR_gain.reshape(dims_vox),
    )


# ---------------------------------------------------------------------------
# Batched patch denoising
# ---------------------------------------------------------------------------


def _denoise_patches(
    patches: torch.Tensor,
    *,
    opt_shrink: bool,
    sigma2: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MP-PCA denoise a batch of (W × M) patches via batched SVD.

    Args:
        patches: (B, W, M) float tensor.
        opt_shrink: Apply Frobenius-optimal singular-value shrinkage.
        sigma2: Known noise variance, or None to estimate via the MP distribution.

    Returns:
        patches_den: (B, W, M) denoised.
        sigma2_b: (B,) noise variance per patch.
        P_b: (B,) signal component count per patch (long).
    """
    B, W, M = patches.shape
    dtype, device = patches.dtype, patches.device

    if min(W, M) == 1:
        return (
            patches.clone(),
            torch.zeros(B, dtype=dtype, device=device),
            torch.ones(B, dtype=torch.long, device=device),
        )

    U, s, Vh = torch.linalg.svd(patches, full_matrices=False)  # (B,W,K), (B,K), (B,K,M)
    K = s.shape[1]
    vals2 = s**2  # (B, K)

    if sigma2 is None:
        P_b, sigma2_b = _mp_estimate(vals2, W, M)
    else:
        sigma2_b = torch.full((B,), sigma2, dtype=dtype, device=device)
        cutoff = sigma2 * (W**0.5 + M**0.5) ** 2
        P_b = (vals2 > cutoff).sum(1).to(torch.long)

    comp_mask = torch.arange(K, device=device).unsqueeze(0) < P_b.unsqueeze(1)  # (B, K)

    if opt_shrink:
        s_den = _opt_shrink_batched(vals2, P_b, sigma2_b, W, M, comp_mask)
    else:
        s_den = s * comp_mask.to(dtype)

    patches_den = (U * s_den.unsqueeze(1)) @ Vh
    return patches_den, sigma2_b, P_b


def _mp_estimate(
    vals2: torch.Tensor, W: int, M: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised Marchenko-Pastur noise estimation for a batch of patches.

    Args:
        vals2: Squared singular values, shape (B, K).
        W: Number of rows (window voxels).
        M: Number of columns (measurements).

    Returns:
        P_b: Signal component counts, shape (B,) long.
        sigma2_b: Noise variance estimates, shape (B,) float.
    """
    B, K = vals2.shape
    device, dtype = vals2.device, vals2.dtype

    P_cands = torch.arange(K, device=device, dtype=dtype)  # (K,)
    cumsum_tail = vals2.flip(1).cumsum(1).flip(1)  # sum of vals²[k:], (B, K)
    denom = ((W - P_cands) * (M - P_cands)).clamp(min=1)  # (K,)
    sigma2_est = cumsum_tail / denom  # (B, K)
    cutoff_est = sigma2_est * (W**0.5 + M**0.5) ** 2  # (B, K)

    # P = index of first singular value that falls below the MP cutoff
    below = vals2 < cutoff_est  # (B, K)
    has_below = below.any(1)  # (B,)
    P_b = torch.where(
        has_below,
        below.long().argmax(1),
        torch.full((B,), K, device=device, dtype=torch.long),
    )

    P_clamped = P_b.clamp(max=K - 1)
    sigma2_b = sigma2_est[torch.arange(B, device=device), P_clamped]
    sigma2_b = torch.where(
        has_below, sigma2_b, torch.zeros(B, dtype=dtype, device=device)
    )
    return P_b, sigma2_b


def _opt_shrink_batched(
    vals2: torch.Tensor,
    P_b: torch.Tensor,
    sigma2_b: torch.Tensor,
    W: int,
    M: int,
    comp_mask: torch.Tensor,
) -> torch.Tensor:
    """Frobenius-optimal singular value shrinkage, batched.

    Signal components are shrunk; noise components are zeroed.

    Returns:
        s_den: Shrunk singular values, shape (B, K).
    """
    M_eff = (W - P_b).clamp(min=1).to(vals2.dtype).unsqueeze(1)  # (B, 1)
    N_eff = (M - P_b).clamp(min=1).to(vals2.dtype).unsqueeze(1)  # (B, 1)
    s2 = sigma2_b.unsqueeze(1)  # (B, 1)

    vals2_safe = vals2.clone()
    vals2_safe[~comp_mask] = 1.0  # avoid div-by-zero on noise components

    shrunk = (
        vals2_safe
        - 2 * (M_eff + N_eff) * s2
        + (M_eff - N_eff) ** 2 * s2**2 / vals2_safe
    )

    cdtype = torch.complex128 if vals2.dtype == torch.float64 else torch.complex64
    s_den = torch.real(torch.sqrt(shrunk.to(cdtype))).to(vals2.dtype)
    s_den = s_den * comp_mask.to(vals2.dtype)
    return s_den


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_inputs(data, window, mask, stride, device):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data.copy())
    if device is None:
        device = data.device if data.is_cuda else torch.device("cpu")
    dtype = torch.float64
    data = data.to(dtype=dtype, device=device)

    window = list(window)
    n_spatial = len(window)
    dims_vox = tuple(data.shape[:n_spatial])

    if mask is None:
        mask = torch.ones(dims_vox, dtype=torch.bool, device=device)
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
