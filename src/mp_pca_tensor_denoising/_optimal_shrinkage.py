"""Frobenius-norm optimal shrinkage for singular values.

Reference: DOI 10.1109/TIT.2017.2653801
"""

import torch


def opt_shrink_frob(
    vals_squared: torch.Tensor, M: int, N: int, sigma2: float
) -> torch.Tensor:
    """Apply Frobenius-norm optimal shrinkage to squared singular values.

    Args:
        vals_squared: Squared singular values, shape (P,).
        M: Number of rows (after potential truncation).
        N: Number of columns (after potential truncation).
        sigma2: Estimated noise variance.

    Returns:
        Shrunk squared singular values, shape (P,).
    """
    # Formula: vals2 = vals2 - 2*(N+M)*sigma2 + (N-M)^2 * sigma2^2 / vals2
    return vals_squared - 2 * (N + M) * sigma2 + (N - M) ** 2 * sigma2**2 / vals_squared
