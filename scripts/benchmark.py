"""Benchmark ``denoise_tensor`` runtime on a synthetic phantom.

Generates a real-valued Shepp-Logan phantom of the requested spatial shape,
broadcasts it across measurement dimensions (adding Gaussian noise), then
runs ``denoise_tensor`` and reports wall-clock time and peak GPU memory.

Usage examples::

    # 1-dir: (64, 64, 32, 5) array
    uv run python scripts/benchmark.py --spatial-shape 64 64 32 --measurements 5

    # Tucker: (64, 64, 32, 2, 2) array
    uv run python scripts/benchmark.py --spatial-shape 64 64 32 --measurements 2 2

    # Large 1-dir run on CPU
    uv run python scripts/benchmark.py --spatial-shape 128 128 64 --measurements 5 --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mppca_denoising import denoise_tensor

# ---------------------------------------------------------------------------
# Shepp-Logan phantom (axis-aligned ellipsoids in normalised [-1,1] space)
# ---------------------------------------------------------------------------

_ELLIPSOIDS = [
    (1.00,  0.00,  0.00,  0.00,  0.6900, 0.9200, 0.90),
    (-0.80,  0.00,  0.00,  0.00,  0.6624, 0.8740, 0.88),
    (-0.20, -0.22,  0.00, -0.25,  0.4100, 0.1600, 0.21),
    (-0.20,  0.22,  0.00, -0.25,  0.3100, 0.1100, 0.22),
    ( 0.10,  0.00,  0.35, -0.25,  0.2100, 0.2500, 0.50),
    ( 0.10,  0.00,  0.10, -0.25,  0.0460, 0.0460, 0.046),
    (-0.01, -0.08, -0.65, -0.25,  0.0460, 0.0230, 0.020),
    (-0.01,  0.06, -0.65, -0.25,  0.0230, 0.0230, 0.020),
    ( 0.01,  0.06, -0.105, 0.625, 0.0460, 0.0230, 0.020),
    ( 0.01,  0.00,  0.100, 0.625, 0.0230, 0.0230, 0.020),
]


def _shepp_logan_3d(shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    for A, x0, y0, z0, a, b, c in _ELLIPSOIDS:
        mask = ((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 + ((Z - z0) / c) ** 2 <= 1
        vol[mask] += A
    return vol.clip(0, None)


# ---------------------------------------------------------------------------

def _get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fmt_mem(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.2f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--spatial-shape", nargs=3, type=int, default=[64, 64, 32],
        metavar=("NX", "NY", "NZ"),
        help="Spatial dimensions (default: 64 64 32)",
    )
    parser.add_argument(
        "--measurements", nargs="+", type=int, default=[5],
        metavar="M",
        help="Measurement dimensions: one int for 1-dir, multiple for Tucker "
             "(default: 5)",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Isotropic sliding-window size (default: 5)",
    )
    parser.add_argument(
        "--opt-shrink", action="store_true", default=True,
        help="Use optimal shrinkage (default: true)",
    )
    parser.add_argument(
        "--no-opt-shrink", dest="opt_shrink", action="store_false",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Patch batch size (default: 4096)",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda | mps | cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--snr", type=float, default=10.0,
        help="Input SNR used to scale noise (default: 10)",
    )
    parser.add_argument(
        "--no-warmup", dest="warmup", action="store_false",
        help="Skip warm-up pass (small patch run)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of timed repetitions (default: 1)",
    )
    args = parser.parse_args()

    device = _get_device(args.device)
    spatial_shape = tuple(args.spatial_shape)
    meas_shape = tuple(args.measurements)
    full_shape = spatial_shape + meas_shape
    window = [args.window] * 3

    print("=" * 60)
    print("denoise_tensor benchmark")
    print("=" * 60)
    print(f"  array shape  : {full_shape}")
    print(f"  window       : {window}")
    print(f"  opt_shrink   : {args.opt_shrink}")
    print(f"  device       : {device}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  repeats      : {args.repeats}")
    print()

    # Build phantom
    rng = np.random.default_rng(1234)
    print(f"  Building phantom {spatial_shape} ...", end=" ", flush=True)
    t0 = time.time()
    phantom = _shepp_logan_3d(spatial_shape)  # (NX, NY, NZ)
    sigma = float(phantom.mean()) / args.snr
    noise = rng.standard_normal(full_shape).astype(np.float32) * sigma
    data = (phantom[..., *([np.newaxis] * len(meas_shape))] + noise).astype(np.float64)
    print(f"done ({time.time() - t0:.1f}s)  dtype={data.dtype}")
    print()

    # Warm-up
    if args.warmup and device != "cpu":
        print("  Warming up ...", end=" ", flush=True)
        small_shape = tuple(min(s, 16) for s in spatial_shape) + meas_shape
        small = data[tuple(slice(0, s) for s in small_shape)]
        denoise_tensor(small, window, device=device, batch_size=args.batch_size,
                       opt_shrink=args.opt_shrink)
        if device == "cuda":
            torch.cuda.synchronize()
        print("done")
        print()

    # Timed run(s)
    times = []
    for rep in range(args.repeats):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        _, _, _, _ = denoise_tensor(
            data, window, device=device, batch_size=args.batch_size,
            opt_shrink=args.opt_shrink,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        times.append(elapsed)
        if args.repeats > 1:
            print(f"  rep {rep + 1}/{args.repeats}: {elapsed:.3f}s")

    print("-" * 60)
    if args.repeats == 1:
        print(f"  Wall time  : {times[0]:.3f}s")
    else:
        print(f"  Wall time  : {min(times):.3f}s min / "
              f"{sum(times)/len(times):.3f}s mean  (over {args.repeats} reps)")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"  Peak VRAM  : {_fmt_mem(peak)}")
    print()


if __name__ == "__main__":
    main()
