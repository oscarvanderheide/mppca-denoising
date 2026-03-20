"""Marchenko-Pastur noise estimation and component selection."""

import torch


def estimate_noise(
    S: torch.Tensor, dims: tuple[int, ...]
) -> tuple[float, int]:
    """Estimate noise variance and number of signal components from singular values.

    Uses the Marchenko-Pastur distribution upper edge as a cutoff: singular
    values below the cutoff are classified as noise.

    Args:
        S: Diagonal matrix from SVD (2D tensor, economy-size), shape (K, K).
        dims: Original dimensions of the tensor being denoised.

    Returns:
        sigma2: Estimated noise variance.
        P: Estimated number of signal components.
    """
    M = S.shape[0]
    N = int(torch.tensor(dims).prod().item()) // M
    vals2 = torch.diag(S) ** 2  # squared singular values

    P_candidates = torch.arange(len(vals2), device=S.device, dtype=S.dtype)
    # sigma2 as a function of number of signal components
    sigma2_estimates = torch.cumsum(vals2.flip(0), dim=0).flip(0) / (M - P_candidates) / (N - P_candidates)
    # upper cutoff of MP distribution
    cutoff_estimates = sigma2_estimates * (M**0.5 + N**0.5) ** 2

    # find first singular value that falls below the cutoff
    below_cutoff = vals2 < cutoff_estimates
    indices = torch.where(below_cutoff)[0]

    if len(indices) == 0:
        # no noise components found
        P = len(vals2)
        sigma2 = 0.0
    else:
        P = int(indices[0].item())  # 0-based: first noise component index = number of signal components
        sigma2 = float(sigma2_estimates[P].item())

    if P == 0 and min(M, N) == 1:
        P = 1

    return sigma2, P


def combined_noise_estimate(
    S_list: list[torch.Tensor],
    dims: tuple[int, ...],
    P_list: list[int],
) -> tuple[float, list[int]]:
    """Combine noise estimates from multiple mode-unfoldings into a single sigma2.

    Args:
        S_list: List of diagonal matrices from SVDs of each mode unfolding.
        dims: Original tensor dimensions.
        P_list: Initial per-mode signal component counts.

    Returns:
        sigma2: Combined noise variance estimate.
        P_list: Updated per-mode signal component counts.
    """
    total_elements = 1
    for d in dims:
        total_elements *= d

    numerator = 0.0
    denominator = 0.0
    for n, (S, P) in enumerate(zip(S_list, P_list)):
        M = S.shape[0]
        N = total_elements // M
        vals2 = torch.diag(S) ** 2
        numerator += float(vals2[P:].sum().item())
        denominator += (M - P) * (N - P)

    sigma2 = numerator / denominator if denominator > 0 else 0.0

    # update P values with combined sigma2
    new_P = []
    for n, S in enumerate(S_list):
        M = S.shape[0]
        N = total_elements // M
        cutoff = sigma2 * (M**0.5 + N**0.5) ** 2
        vals2 = torch.diag(S) ** 2
        new_P.append(int((vals2 > cutoff).sum().item()))

    return sigma2, new_P


def discard_noise_components(
    U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, sigma2: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Discard noise singular components based on MP cutoff.

    Args:
        U: Left singular vectors, shape (M, K).
        S: Diagonal matrix, shape (K, K).
        V: Right singular vectors, shape (N, K).
        sigma2: Noise variance.

    Returns:
        U, S, V truncated to P components, and P.
    """
    M = U.shape[0]
    N = V.shape[0]
    cutoff = sigma2 * (M**0.5 + N**0.5) ** 2
    vals2 = torch.diag(S) ** 2
    P = int((vals2 > cutoff).sum().item())
    return U[:, :P], S[:P, :P], V[:, :P], P
