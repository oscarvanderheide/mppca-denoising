"""Shared helpers for MATLAB/Python comparison workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

# Each row: (A, x0, y0, z0, a, b, c) in normalised [-1, 1] coordinates.
_ELLIPSOIDS = [
    (1.00, 0.00, 0.00, 0.00, 0.69, 0.92, 0.90),
    (-0.80, 0.00, 0.00, 0.00, 0.6624, 0.874, 0.88),
    (-0.20, -0.22, 0.00, -0.25, 0.41, 0.16, 0.21),
    (-0.20, 0.22, 0.00, -0.25, 0.31, 0.11, 0.22),
    (0.10, 0.00, 0.35, -0.25, 0.21, 0.25, 0.50),
    (0.10, 0.00, 0.10, -0.25, 0.046, 0.046, 0.046),
    (-0.01, -0.08, -0.65, -0.25, 0.046, 0.023, 0.020),
    (-0.01, 0.06, -0.65, -0.25, 0.023, 0.023, 0.020),
    (0.01, 0.06, -0.105, 0.625, 0.046, 0.023, 0.020),
    (0.01, 0.00, 0.100, 0.625, 0.023, 0.023, 0.020),
]

DEFAULT_SPATIAL_SHAPE = (48, 48, 24)
DEFAULT_WINDOW = (5, 5, 5)
DEFAULT_CASES = (
    {"name": "case_1dir", "measurement_shape": (8,)},
    {"name": "case_2dir", "measurement_shape": (4, 3)},
)


def shepp_logan_3d(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a real-valued 3-D Shepp-Logan phantom."""
    nx, ny, nz = shape
    xs = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    zs = np.linspace(-1.0, 1.0, nz, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    phantom = np.zeros(shape, dtype=np.float64)
    for amp, x0, y0, z0, a, b, c in _ELLIPSOIDS:
        inside = (
            ((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 + ((Z - z0) / c) ** 2
        ) <= 1.0
        phantom[inside] += amp
    return phantom.clip(0.0, None)


def build_signal(
    spatial_shape: tuple[int, int, int],
    measurement_shape: tuple[int, ...],
) -> np.ndarray:
    """Build a low-rank synthetic signal with explicit measurement structure."""
    phantom = shepp_logan_3d(spatial_shape)
    xs = np.linspace(-1.0, 1.0, spatial_shape[0], dtype=np.float64)[:, None, None]
    ys = np.linspace(-1.0, 1.0, spatial_shape[1], dtype=np.float64)[None, :, None]
    zs = np.linspace(-1.0, 1.0, spatial_shape[2], dtype=np.float64)[None, None, :]

    basis0 = phantom
    basis1 = phantom * (0.75 + 0.25 * xs)
    basis2 = phantom * (0.65 + 0.20 * ys**2 + 0.15 * zs)

    if len(measurement_shape) == 1:
        (m0,) = measurement_shape
        theta = np.linspace(0.0, 2.0 * np.pi, m0, endpoint=False, dtype=np.float64)
        coeff0 = 1.0 + 0.18 * np.sin(theta)
        coeff1 = 0.25 * np.cos(2.0 * theta - 0.3)
        coeff2 = 0.15 * np.sin(3.0 * theta + 0.2)
        signal = (
            basis0[..., None] * coeff0
            + basis1[..., None] * coeff1
            + basis2[..., None] * coeff2
        )
        return signal.astype(np.float64)

    if len(measurement_shape) == 2:
        m0, m1 = measurement_shape
        theta0 = np.linspace(0.0, 2.0 * np.pi, m0, endpoint=False, dtype=np.float64)
        theta1 = np.linspace(0.0, 2.0 * np.pi, m1, endpoint=False, dtype=np.float64)
        coeff0 = np.outer(1.0 + 0.12 * np.sin(theta0), 1.0 + 0.10 * np.cos(theta1))
        coeff1 = np.outer(np.cos(2.0 * theta0 - 0.1), np.sin(theta1 + 0.4))
        coeff2 = np.outer(np.sin(theta0 - 0.5), np.cos(2.0 * theta1 - 0.2))
        signal = (
            basis0[..., None, None] * coeff0
            + 0.20 * basis1[..., None, None] * coeff1
            + 0.12 * basis2[..., None, None] * coeff2
        )
        return signal.astype(np.float64)

    raise ValueError(f"Unsupported measurement shape: {measurement_shape}")


def make_noisy_case(
    spatial_shape: tuple[int, int, int],
    measurement_shape: tuple[int, ...],
    *,
    noise_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return clean/noisy synthetic data and metadata for one case."""
    clean = build_signal(spatial_shape, measurement_shape)
    noise_sigma = float(clean.std() * noise_fraction)
    noise = rng.standard_normal(clean.shape, dtype=np.float64) * noise_sigma
    noisy = clean + noise
    return {
        "signal_clean": clean,
        "signal_noisy": noisy,
        "noise_sigma": noise_sigma,
        "signal_mean": float(clean.mean()),
        "signal_std": float(clean.std()),
    }


def ensure_output_layout(output_dir: Path) -> None:
    """Create the standard output directories for the workflow."""
    for name in ("inputs", "matlab", "python", "reports"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def save_mat(path: Path, **arrays: Any) -> None:
    """Save arrays to a MATLAB v5 .mat file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(path, arrays, do_compression=True)


def load_mat_array(path: Path, key: str | None = None) -> np.ndarray:
    """Load one array from a MATLAB .mat file (v5 or v7.3)."""
    try:
        mat = sio.loadmat(str(path))
    except NotImplementedError:
        import h5py

        with h5py.File(str(path), "r") as handle:
            if key is None:
                key = next(name for name in handle.keys() if not name.startswith("#"))
            return np.array(handle[key]).T

    if key is None:
        key = next(name for name in mat if not name.startswith("_"))
    return np.array(mat[key])


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON document with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def to_builtin(value: Any) -> Any:
    """Convert NumPy values into JSON-serialisable builtins."""
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def relative_to(path: Path, root: Path) -> str:
    """Return a forward-slash relative path string."""
    return path.relative_to(root).as_posix()
