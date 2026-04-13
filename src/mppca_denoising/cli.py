from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch

from .denoise import denoise_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MP-PCA tensor denoising on an input array file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input array file (.npy, .npz, or .mat)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file (.npz or .mat). Defaults to <input_stem>_denoised.npz "
            "next to the input file."
        ),
    )
    parser.add_argument(
        "--window",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Sliding-window size per spatial dimension, e.g. --window 5 5 5",
    )
    parser.add_argument(
        "--input-key",
        default=None,
        help="Array name to load from a .npz or .mat input file",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional mask array file (.npy, .npz, or .mat)",
    )
    parser.add_argument(
        "--mask-key",
        default=None,
        help="Array name to load from a .npz or .mat mask file",
    )
    parser.add_argument(
        "--stride",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Stride per spatial dimension (default: 1 in each dim)",
    )
    parser.add_argument(
        "--center-assign",
        action="store_true",
        help="Assign only the centre voxel from each denoised patch",
    )
    parser.add_argument(
        "--no-opt-shrink",
        dest="opt_shrink",
        action="store_false",
        help="Disable optimal singular-value shrinkage",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=None,
        help="Known noise variance. If omitted, it is estimated from the data.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, e.g. cpu, cuda, cuda:0, mps",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "complex64", "complex128"),
        default=None,
        help="Torch dtype to use for computation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Patch batch size passed to denoise_tensor",
    )
    parser.set_defaults(opt_shrink=True)
    return parser.parse_args()


def _load_array(path: Path, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        if key is not None:
            raise ValueError("--input-key/--mask-key is not valid for .npy files")
        return np.load(path)

    if suffix == ".npz":
        with np.load(path) as archive:
            if key is None:
                names = archive.files
                if len(names) != 1:
                    available = ", ".join(names)
                    raise ValueError(
                        f"{path} contains multiple arrays ({available}); pass a key"
                    )
                key = names[0]
            if key not in archive:
                available = ", ".join(archive.files)
                raise KeyError(f"Key {key!r} not found in {path}; available: {available}")
            return archive[key]

    if suffix == ".mat":
        data = sio.loadmat(path)
        if key is None:
            names = sorted(name for name in data if not name.startswith("_"))
            if len(names) != 1:
                available = ", ".join(names)
                raise ValueError(
                    f"{path} contains multiple arrays ({available}); pass a key"
                )
            key = names[0]
        if key not in data:
            names = sorted(name for name in data if not name.startswith("_"))
            available = ", ".join(names)
            raise KeyError(f"Key {key!r} not found in {path}; available: {available}")
        return np.asarray(data[key])

    raise ValueError(f"Unsupported file format for {path}")


def _save_output(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        np.savez_compressed(path, **arrays)
        return

    if suffix == ".mat":
        sio.savemat(path, arrays, do_compression=True)
        return

    raise ValueError("Output path must end in .npz or .mat")


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_denoised.npz")


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return getattr(torch, name)


def _get_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalise_window_and_stride(
    data: np.ndarray,
    window: list[int],
    stride: list[int] | None,
) -> tuple[list[int], list[int] | None]:
    if len(window) > data.ndim:
        raise ValueError(
            f"Window has {len(window)} dimensions but input data has only {data.ndim} dimensions"
        )

    if len(window) == 1 and data.ndim > 2:
        raise ValueError(
            "A single-value window is ambiguous for multidimensional input. "
            "For typical MRI data shaped like (X, Y, Z, M), pass all spatial dimensions "
            "explicitly, e.g. --window 5 5 5."
        )

    if stride is None:
        return window, None

    if len(stride) == 1 and len(window) > 1:
        stride = stride * len(window)

    if len(stride) != len(window):
        raise ValueError(
            f"Stride has {len(stride)} dimensions but window has {len(window)} dimensions"
        )

    return window, stride


def main() -> None:
    args = parse_args()

    print(f"Loading input from {args.input}", flush=True)
    data = _load_array(args.input, args.input_key)
    mask = _load_array(args.mask, args.mask_key) if args.mask is not None else None
    window, stride = _normalise_window_and_stride(data, args.window, args.stride)
    device = _get_device(args.device)
    output_path = args.output or _default_output_path(args.input)

    print(
        f"Input shape: {data.shape}, dtype: {data.dtype}",
        flush=True,
    )
    print(
        f"Using device={device}, window={window}, stride={stride or [1] * len(window)}",
        flush=True,
    )
    print(f"Using batch_size={args.batch_size}", flush=True)
    print(f"Writing output to {output_path}", flush=True)
    print("Starting denoising...", flush=True)

    try:
        denoised, sigma2_map, n_signal_map, snr_gain_map = denoise_tensor(
            data,
            window=window,
            mask=mask,
            stride=stride,
            center_assign=args.center_assign,
            opt_shrink=args.opt_shrink,
            sigma2=args.sigma2,
            device=device,
            dtype=_torch_dtype(args.dtype),
            batch_size=args.batch_size,
        )
    except torch.OutOfMemoryError as exc:
        if device.startswith("cuda"):
            raise SystemExit(
                "CUDA ran out of memory while extracting or denoising patches. "
                "Retry with a smaller batch size, for example: --batch-size 1024"
            ) from exc
        raise

    arrays = {
        "denoised": denoised.detach().cpu().numpy(),
        "sigma2": sigma2_map.detach().cpu().numpy(),
        "p": n_signal_map.detach().cpu().numpy(),
        "snr_gain": snr_gain_map.detach().cpu().numpy(),
    }
    _save_output(output_path, arrays)
    print(f"Saved denoising outputs to {output_path}")


if __name__ == "__main__":
    main()