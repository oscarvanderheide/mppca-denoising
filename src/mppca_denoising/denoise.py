"""MP-PCA denoising of MRI data.

Implements the sliding-window tensor denoising (tMPPCA) from:
    Olesen et al., "Tensor denoising of multidimensional MRI data",
    Magn Reson Med, 2022. doi:10.1002/mrm.29478

For data with shape (*spatial_dims, M) the algorithm treats M as a single
measurement mode, which is equivalent to plain MPPCA.  For data with multiple
non-spatial dimensions, e.g. (*spatial, M1, M2), the Tucker sequential
mode-unfolding decomposition is used: each mode is denoised in sequence
(tMPPCA), which can give tighter rank estimates than treating all measurements
as a single flat index.
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
    """Denoise MRI data of shape (*spatial_dims, M) using tMPPCA.

    Args:
        data:          Shape (*spatial_dims, *meas_dims). First len(window) dims are spatial.
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
        sigma2_map:   Noise variance per voxel, shape (*spatial_dims,).
        n_signal_map: Signal component count per voxel, shape (*spatial_dims,).
        snr_gain_map: Estimated SNR improvement per voxel, shape (*spatial_dims,).
    """
    # Large batches keep the GPU fully occupied.
    if _is_cuda(device):
        batch_size = max(batch_size, 262144)

    # Validate inputs and move to target device and dtype.
    data, window, mask, stride, device, dtype = _prepare_inputs(
        data, window, mask, stride, device, dtype
    )
    n_spatial = len(window)
    spatial_shape = data.shape[:n_spatial]
    meas_shape = tuple(data.shape[n_spatial:])  # shape of per-voxel measurement tensor
    n_meas = prod(meas_shape)  # total measurements per voxel
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
            patches, meas_shape=meas_shape, opt_shrink=opt_shrink, sigma2=sigma2
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
# Batched patch denoising — Tucker sequential mode unfolding (tMPPCA)
# ---------------------------------------------------------------------------


def _denoise_patches(
    patches: torch.Tensor,
    *,
    meas_shape: tuple[int, ...],
    opt_shrink: bool,
    sigma2: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise a batch of patches via MP-PCA or Tucker sequential mode unfolding.

    For single measurement mode (len(meas_shape)==1) this uses plain matrix
    MP-PCA on the (B, W, M) matrix, matching the MATLAB reference exactly.

    For multiple measurement modes the Tucker sequential (tMPPCA) algorithm is
    used: unfold and truncate along each mode in sequence with a shared sigma2.

    Args:
        patches:    (B, W, n_meas) float or complex tensor.
        meas_shape: Per-voxel measurement shape; prod(meas_shape) == n_meas.
        opt_shrink: Apply Frobenius-optimal shrinkage on the final mode.
        sigma2:     Known noise variance, or None to estimate from data.

    Returns:
        denoised_patches: (B, W, n_meas).
        sigma2_b:         (B,) noise variance per patch.
        n_signal_b:       (B,) signal rank (used in SNR formula).
    """
    B, W, n_meas = patches.shape
    device = patches.device
    real_dtype = patches.real.dtype if patches.is_complex() else patches.dtype

    # ------------------------------------------------------------------
    # Single measurement-mode fast path: plain matrix MP-PCA.
    # This exactly matches the MATLAB denoise_recursive_tensor behaviour
    # for 4-D data (spatial + 1 measurement dimension).
    # ------------------------------------------------------------------
    if len(meas_shape) == 1:
        M = n_meas
        # Gram matrix on the smaller dimension.
        if W >= M:
            gram = patches.mH @ patches  # (B, M, M)
        else:
            gram = patches @ patches.mH  # (B, W, W)
        K = min(W, M)
        sq_sv, evecs = torch.linalg.eigh(gram)  # ascending
        sq_sv = sq_sv.flip(-1).clamp(min=0)[:, :K]  # descending (B, K)
        evecs = evecs.flip(-1)[:, :, :K]  # (B, min_dim, K)

        if sigma2 is None:
            n_signal_b, sigma2_b = _mp_estimate(sq_sv, W, M)
        else:
            sigma2_b = torch.full((B,), sigma2, dtype=real_dtype, device=device)
            cutoff = sigma2_b * (W**0.5 + M**0.5) ** 2
            n_signal_b = (sq_sv > cutoff.unsqueeze(1)).sum(1).to(torch.long)

        max_P = int(n_signal_b.max().item())
        if max_P == 0:
            denoised = torch.zeros_like(patches)
            return denoised, sigma2_b, n_signal_b

        # Zero-mask components beyond each patch's P.
        keep = torch.arange(max_P, device=device).unsqueeze(0) < n_signal_b.unsqueeze(
            1
        )  # (B, max_P)

        if W >= M:
            # evecs are V (right singular vectors), shape (B, M, K).
            V_P = evecs[:, :, :max_P]  # (B, M, max_P)
            if opt_shrink:
                # Frobenius-optimal shrinkage on the signal singular values.
                signal_mask = keep  # (B, max_P)
                shrunk = _opt_shrink_batched(
                    sq_sv[:, :max_P], n_signal_b, sigma2_b, W, M, signal_mask
                )  # (B, max_P) shrunk singular values
                orig_sv = (
                    sq_sv[:, :max_P].sqrt().masked_fill(sq_sv[:, :max_P] == 0, 1.0)
                )
                scale = (shrunk / orig_sv) * keep.to(sq_sv.dtype)  # (B, max_P)
                # X_rec = patches @ V_P @ diag(scale) @ V_P.H
                XV = patches @ V_P  # (B, W, max_P)
                denoised = (XV * scale.unsqueeze(1)) @ V_P.mH  # (B, W, M)
            else:
                V_P_masked = V_P * keep.to(V_P.dtype).unsqueeze(1)  # (B, M, max_P)
                denoised = patches @ (V_P_masked @ V_P_masked.mH)  # (B, W, M)
        else:
            # evecs are U (left singular vectors), shape (B, W, K).
            U_P = evecs[:, :, :max_P]  # (B, W, max_P)
            if opt_shrink:
                signal_mask = keep
                shrunk = _opt_shrink_batched(
                    sq_sv[:, :max_P], n_signal_b, sigma2_b, W, M, signal_mask
                )
                orig_sv = (
                    sq_sv[:, :max_P].sqrt().masked_fill(sq_sv[:, :max_P] == 0, 1.0)
                )
                scale = (shrunk / orig_sv) * keep.to(sq_sv.dtype)
                UX = U_P.mH @ patches  # (B, max_P, M)
                denoised = (U_P * scale.unsqueeze(2)) @ UX  # (B, W, M)
            else:
                U_P_masked = U_P * keep.to(U_P.dtype).unsqueeze(2)
                denoised = (U_P_masked @ U_P_masked.mH) @ patches  # (B, W, M)

        return denoised, sigma2_b, n_signal_b

    # ------------------------------------------------------------------
    # Multi measurement-mode path: Tucker sequential (tMPPCA).
    # ------------------------------------------------------------------
    # dims[0] = W (spatial), dims[1..] = meas_shape.
    dims = (W,) + meas_shape
    num_modes = len(dims)

    # Reshape to multi-mode tensor (B, W, M1, M2, ...).
    X = patches.reshape((B,) + dims)

    # ------------------------------------------------------------------
    # Pass 1 — estimate a single shared sigma2 from all mode unfoldings.
    # For each mode n: unfold X₀ to (B, dims[n], rest), compute eigh,
    # get initial noise estimate.  Then combine all estimates into one sigma2.
    # ------------------------------------------------------------------
    if sigma2 is None:
        modal_sv: list[
            tuple[torch.Tensor, int, int]
        ] = []  # (sq_singvals, M, N) per mode
        for n in range(num_modes):
            Xn, Mn, Nn = _mode_unfold(X, n, dims)
            sq_sv = _eigvalsh_descending(Xn, Mn, Nn)
            modal_sv.append((sq_sv, Mn, Nn))

        # Initial per-mode P via individual MP estimates.
        P_init = [_mp_estimate(sv, Mn, Nn)[0] for sv, Mn, Nn in modal_sv]

        # Combined sigma2: weighted average of per-mode noise tail energies.
        # σ² = Σ_n tail_n / Σ_n (Mn-Pn)*(Nn-Pn)   (MATLAB: combined_noise_estimate)
        numer = torch.zeros(B, dtype=real_dtype, device=device)
        denom = torch.zeros(B, dtype=real_dtype, device=device)
        for (sq_sv, Mn, Nn), P_n in zip(modal_sv, P_init):
            # Tail energy: sum of eigenvalues that are noise (index >= P_n).
            signal_mask = torch.arange(sq_sv.shape[1], device=device).unsqueeze(
                0
            ) < P_n.unsqueeze(1)
            tail = sq_sv.masked_fill(signal_mask, 0.0).sum(1)
            numer += tail
            denom += ((Mn - P_n) * (Nn - P_n)).clamp(min=1).to(real_dtype)
        sigma2_b = numer / denom.clamp(min=1)

        # Refine P estimates using the combined sigma2.
        P_all = []
        for sq_sv, Mn, Nn in modal_sv:
            cutoff = sigma2_b * (Mn**0.5 + Nn**0.5) ** 2  # (B,)
            P_n = (sq_sv > cutoff.unsqueeze(1)).sum(1).to(torch.long)
            P_all.append(P_n)
    else:
        sigma2_b = torch.full((B,), sigma2, dtype=real_dtype, device=device)
        P_all = []
        for n in range(num_modes):
            Xn, Mn, Nn = _mode_unfold(X, n, dims)
            sq_sv = _eigvalsh_descending(Xn, Mn, Nn)
            cutoff = sigma2_b * (Mn**0.5 + Nn**0.5) ** 2
            P_all.append((sq_sv > cutoff.unsqueeze(1)).sum(1).to(torch.long))

    # ------------------------------------------------------------------
    # Pass 2 — sequential Tucker truncation (forward sweep).
    # After mode n: X (unfolded) ← U_n^H @ X_n  (the "passed-forward" part).
    # The Tucker core X shrinks along mode n from dims[n] to P_n.
    # ------------------------------------------------------------------
    cur_dims = list(dims)  # tracks current (possibly reduced) size per mode
    U_list: list[tuple[torch.Tensor, int]] = []  # (U_n, orig_Mn) per mode

    for n in range(num_modes):
        P_n = P_all[n]
        max_P = int(P_n.max().item())

        Xn, Mn, Nn = _mode_unfold(X, n, cur_dims)

        if max_P == 0 or Mn == 0 or Nn == 0:
            # All patches are pure noise at this mode.
            X = torch.zeros_like(X)
            U_list.append(
                (torch.zeros(B, Mn, 0, dtype=patches.dtype, device=device), Mn)
            )
            cur_dims[n] = 0
            break

        # left singular vectors U_n of X_n via eigh of the smaller gram matrix.
        U_n = _left_singular_vecs(Xn, Mn, Nn, max_P)  # (B, Mn, max_P)

        # Project: Xn_reduced = U_n^H @ Xn  →  (B, max_P, Nn).
        # This is what the MATLAB passes as "X = V*S" to the next iteration.
        Xn_reduced = U_n.mH @ Xn  # (B, max_P, Nn)

        # Zero out components beyond each patch's own P_n.
        keep = torch.arange(max_P, device=device).unsqueeze(0) < P_n.unsqueeze(
            1
        )  # (B, max_P)
        Xn_reduced = Xn_reduced * keep.to(Xn_reduced.dtype).unsqueeze(2)

        U_list.append((U_n, Mn))
        cur_dims[n] = max_P

        # Refold back to multi-mode shape with updated dim n.
        rest = [cur_dims[i] for i in range(num_modes) if i != n]
        X = _mode_refold(Xn_reduced.reshape([B, max_P] + rest), n, cur_dims, num_modes)

    # ------------------------------------------------------------------
    # Optional: Frobenius-optimal shrinkage on the final Tucker core,
    # applied to the last mode (mirrors MATLAB's last-iteration shrinkage).
    # ------------------------------------------------------------------
    if opt_shrink and len(P_all) > 0 and all(d > 0 for d in cur_dims):
        n_last = num_modes - 1
        Xl, Ml, Nl = _mode_unfold(X, n_last, cur_dims)
        Kl = min(Ml, Nl)
        if Kl > 1:
            P_last = P_all[n_last]
            sq_sv_l, evecs_l = _eigh_descending(Xl, Ml, Nl)
            signal_mask_l = torch.arange(Kl, device=device).unsqueeze(
                0
            ) < P_last.unsqueeze(1)
            shrunk_l = _opt_shrink_batched(
                sq_sv_l, P_last, sigma2_b, Ml, Nl, signal_mask_l
            )
            orig_sv_l = sq_sv_l.sqrt()
            scale_l = shrunk_l / orig_sv_l.masked_fill(orig_sv_l == 0, 1.0)  # (B, Kl)

            # Reconstruct Xl with scaled singular values:
            # Xl_out = U_l @ diag(scale) @ U_l^H @ Xl   (if Ml <= Nl, gram is on left)
            if Ml <= Nl:
                U_l = evecs_l  # (B, Ml, Kl) left vecs
                XV = U_l.mH @ Xl  # (B, Kl, Nl)
                Xl_out = (U_l * scale_l.unsqueeze(1)) @ XV  # (B, Ml, Nl)
            else:
                V_l = evecs_l  # (B, Nl, Kl) right vecs
                XV = Xl @ V_l  # (B, Ml, Kl)
                Xl_out = (XV * scale_l.unsqueeze(1)) @ V_l.mH  # (B, Ml, Nl)

            rest_l = [cur_dims[i] for i in range(num_modes) if i != n_last]
            X = _mode_refold(
                Xl_out.reshape([B, Ml] + rest_l), n_last, cur_dims, num_modes
            )

    # ------------------------------------------------------------------
    # Backward pass — reconstruct full-size patch from Tucker core.
    # X is the compressed Tucker core (B, P0, P1, ...).
    # Multiply back by each U factor in reverse order.
    # ------------------------------------------------------------------
    for n in reversed(range(len(U_list))):
        U_n, orig_Mn = U_list[n]
        max_Pn = U_n.shape[2]

        if max_Pn == 0:
            # Zero mode: reconstruct zero array with original mode size.
            cur_dims[n] = orig_Mn
            rest = [cur_dims[i] for i in range(num_modes) if i != n]
            zero_shape = [B, orig_Mn] + rest
            X = _mode_refold(
                torch.zeros(zero_shape, dtype=patches.dtype, device=device),
                n,
                cur_dims,
                num_modes,
            )
            continue

        # Unfold core along mode n using current (reduced) dims: (B, max_Pn, rest).
        Xcore, _, _ = _mode_unfold(X, n, cur_dims)

        # Xout = U_n @ Xcore  →  (B, orig_Mn, rest).
        Xout = U_n @ Xcore  # (B, orig_Mn, prod(rest))
        cur_dims[n] = orig_Mn
        rest = [cur_dims[i] for i in range(num_modes) if i != n]
        X = _mode_refold(Xout.reshape([B, orig_Mn] + rest), n, cur_dims, num_modes)

    denoised_patches = X.reshape(B, W, n_meas)
    n_signal_b = P_all[0]  # spatial-mode rank for SNR formula
    return denoised_patches, sigma2_b, n_signal_b


# ---------------------------------------------------------------------------
# Tucker mode operations
# ---------------------------------------------------------------------------


def _mode_unfold(
    X: torch.Tensor,
    n: int,
    dims: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, int, int]:
    """Mode-n unfolding: (B, d0, d1, ...) → (B, dims[n], prod(other dims)).

    Returns (Xn, Mn, Nn) where Mn = dims[n] and Nn = prod(all other dims).
    """
    num_modes = len(dims)
    # Bring axis n+1 (the n-th mode, since axis 0 is batch) to position 1.
    perm = [0, n + 1] + [i for i in range(1, num_modes + 1) if i != n + 1]
    Xn = X.permute(perm).reshape(X.shape[0], dims[n], -1)
    return Xn, dims[n], Xn.shape[2]


def _mode_refold(
    Xn: torch.Tensor,
    n: int,
    cur_dims: list[int],
    num_modes: int,
) -> torch.Tensor:
    """Inverse of _mode_unfold: (B, cur_dims[n], rest_sizes...) → (B, d0, d1...).

    cur_dims must already reflect the updated size at mode n.
    """
    B = Xn.shape[0]
    perm_fwd = [0, n + 1] + [i for i in range(1, num_modes + 1) if i != n + 1]
    perm_inv = [0] + [perm_fwd.index(i) for i in range(1, num_modes + 1)]
    # Shape after forward permutation: [B, cur_dims[perm_fwd[1]-1], cur_dims[perm_fwd[2]-1], ...]
    shape_perm = [B] + [cur_dims[perm_fwd[i + 1] - 1] for i in range(num_modes)]
    return Xn.reshape(shape_perm).permute(perm_inv)


def _eigvalsh_descending(
    Xn: torch.Tensor,
    Mn: int,
    Nn: int,
) -> torch.Tensor:
    """Eigenvalues only (no eigenvectors) of the smaller gram matrix of Xn.

    Faster than _eigh_descending when eigenvectors are not needed.

    Returns sq_singvals: (B, K) squared singular values in descending order.
    """
    K = min(Mn, Nn)
    if Mn <= Nn:
        gram = Xn @ Xn.mH  # (B, Mn, Mn)
    else:
        gram = Xn.mH @ Xn  # (B, Nn, Nn)
    sq_sv = torch.linalg.eigvalsh(gram)
    return sq_sv.flip(-1).clamp(min=0)[:, :K]


def _eigh_descending(
    Xn: torch.Tensor,
    Mn: int,
    Nn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of the smaller gram matrix of Xn (B, Mn, Nn).

    Returns (sq_singvals, evecs) in **descending** order.
    sq_singvals: (B, K) squared singular values, K = min(Mn, Nn).
    evecs:       (B, min_dim, K) eigenvectors of the gram matrix used.
    """
    K = min(Mn, Nn)
    if Mn <= Nn:
        gram = Xn @ Xn.mH  # (B, Mn, Mn)
        sq_sv, evecs = torch.linalg.eigh(gram)
    else:
        gram = Xn.mH @ Xn  # (B, Nn, Nn)
        sq_sv, evecs = torch.linalg.eigh(gram)
    sq_sv = sq_sv.flip(-1).clamp(min=0)[:, :K]
    evecs = evecs.flip(-1)[:, :, :K]
    return sq_sv, evecs


def _left_singular_vecs(
    Xn: torch.Tensor,
    Mn: int,
    Nn: int,
    max_P: int,
) -> torch.Tensor:
    """Top-max_P left singular vectors of Xn (B, Mn, Nn).

    When Mn <= Nn: left vecs = eigenvectors of Xn @ Xn^H (exact).
    When Mn > Nn:  compute right vecs via Xn^H @ Xn, then derive left vecs
                   by normalising Xn @ V (columns are already orthogonal).

    Returns U: (B, Mn, max_P).
    """
    if Mn <= Nn:
        gram = Xn @ Xn.mH  # (B, Mn, Mn)
        _, evecs = torch.linalg.eigh(gram)
        U = evecs.flip(-1)[:, :, :max_P]
    else:
        gram = Xn.mH @ Xn  # (B, Nn, Nn)
        _, evecs = torch.linalg.eigh(gram)
        V = evecs.flip(-1)[:, :, :max_P]  # (B, Nn, max_P) right singular vecs
        US = Xn @ V  # (B, Mn, max_P) = U * Σ  (columns are orthogonal)
        # Columns of US are already orthogonal (U*Σ), just normalise to get U.
        norms = US.norm(dim=1, keepdim=True).clamp(min=1e-30)
        U = US / norms
    return U


# ---------------------------------------------------------------------------
# MP noise estimation and shrinkage
# ---------------------------------------------------------------------------


def _mp_estimate(
    sq_singvals: torch.Tensor, M: int, N: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate signal rank and noise variance via the Marchenko-Pastur law.

    For each candidate split P, the implied noise variance is derived from the
    tail eigenvalue sum.  The smallest P whose P-th eigenvalue still exceeds the
    corresponding MP upper edge is returned as the signal rank.

    Args:
        sq_singvals: (B, K) squared singular values, descending, K = min(M, N).
        M, N:        Matrix dimensions.

    Returns:
        n_signal: (B,) long  — signal component count per patch.
        sigma2_b: (B,) float — noise variance estimate per patch.
    """
    B, K = sq_singvals.shape
    device, dtype = sq_singvals.device, sq_singvals.dtype

    # Tail energy: sum of sq_singvals[k:] for each candidate split index k.
    tail_energy = sq_singvals.flip(1).cumsum(1).flip(1)  # (B, K)

    # Implied noise variance for each candidate P: σ² = Σλ[P:] / ((M-P)*(N-P)).
    p_range = torch.arange(K, device=device, dtype=dtype)
    noise_denom = ((M - p_range) * (N - p_range)).clamp(min=1)
    sigma2_cands = tail_energy / noise_denom  # (B, K)

    # MP upper edge: λ ≤ σ² * (√M + √N)².
    mp_cutoff = sigma2_cands * (M**0.5 + N**0.5) ** 2  # (B, K)

    # The first eigenvalue that falls below the cutoff marks where noise starts.
    below_cutoff = sq_singvals < mp_cutoff  # (B, K)
    any_below = below_cutoff.any(1)  # (B,)
    n_signal = torch.where(
        any_below,
        below_cutoff.long().argmax(1),
        torch.full((B,), K, device=device, dtype=torch.long),
    )

    # Read off the noise variance at the estimated rank.
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

    # Convert to torch tensor, move to device and change dtype if needed.
    # We copy if already a numpy array to avoid sharing memory with the caller.
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
