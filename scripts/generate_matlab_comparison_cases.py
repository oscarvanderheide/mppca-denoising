"""Generate deterministic phantom inputs for MATLAB/Python comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from matlab_compare_common import (
    DEFAULT_CASES,
    DEFAULT_SPATIAL_SHAPE,
    DEFAULT_WINDOW,
    ensure_output_layout,
    make_noisy_case,
    relative_to,
    save_mat,
    write_json,
)
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug") / "matlab_compare",
        help="Directory where manifest, inputs, and results are stored",
    )
    parser.add_argument(
        "--spatial-shape",
        type=int,
        nargs=3,
        default=DEFAULT_SPATIAL_SHAPE,
        metavar=("NX", "NY", "NZ"),
        help="3-D phantom shape (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=3,
        default=DEFAULT_WINDOW,
        metavar=("WX", "WY", "WZ"),
        help="Sliding window passed to MATLAB and Python (default: %(default)s)",
    )
    parser.add_argument(
        "--single-measurements",
        type=int,
        default=DEFAULT_CASES[0]["measurement_shape"][0],
        help="Measurement count for the single-measurement-axis case",
    )
    parser.add_argument(
        "--tensor-measurements",
        type=int,
        nargs=2,
        default=DEFAULT_CASES[1]["measurement_shape"],
        metavar=("M1", "M2"),
        help="Measurement shape for the two-measurement-axis case",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.08,
        help="Noise sigma as a fraction of the clean signal std (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for noise generation (default: %(default)s)",
    )
    parser.add_argument(
        "--opt-shrink",
        dest="opt_shrink",
        action="store_true",
        help="Use Frobenius-optimal shrinkage in both MATLAB and Python",
    )
    parser.add_argument(
        "--no-opt-shrink",
        dest="opt_shrink",
        action="store_false",
        help="Disable Frobenius-optimal shrinkage in both MATLAB and Python",
    )
    parser.set_defaults(opt_shrink=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_layout(output_dir)

    rng = np.random.default_rng(args.seed)
    cases = [
        {"name": "case_1dir", "measurement_shape": (args.single_measurements,)},
        {"name": "case_2dir", "measurement_shape": tuple(args.tensor_measurements)},
    ]

    manifest_cases: list[dict[str, object]] = []
    print("=" * 72)
    print("Generating MATLAB comparison cases")
    print("=" * 72)
    print(f"Output directory : {output_dir}")
    print(f"Spatial shape    : {tuple(args.spatial_shape)}")
    print(f"Window           : {tuple(args.window)}")
    print(f"Noise fraction   : {args.noise_fraction}")
    print(f"Seed             : {args.seed}")
    print(f"Opt shrink       : {args.opt_shrink}")
    print()

    for case in cases:
        name = str(case["name"])
        measurement_shape = tuple(int(v) for v in case["measurement_shape"])
        payload = make_noisy_case(
            tuple(args.spatial_shape),
            measurement_shape,
            noise_fraction=args.noise_fraction,
            rng=rng,
        )
        input_path = output_dir / "inputs" / f"{name}_input.mat"
        save_mat(
            input_path,
            signal_clean=payload["signal_clean"],
            signal_noisy=payload["signal_noisy"],
        )

        manifest_cases.append(
            {
                "name": name,
                "measurement_shape": list(measurement_shape),
                "input_mat": relative_to(input_path, output_dir),
                "matlab_output_mat": f"matlab/{name}_result.mat",
                "matlab_runtime_json": f"matlab/{name}_runtime.json",
                "python_output_mat": f"python/{name}_result.mat",
                "python_runtime_json": f"python/{name}_runtime.json",
                "noise_sigma": payload["noise_sigma"],
                "signal_mean": payload["signal_mean"],
                "signal_std": payload["signal_std"],
            }
        )

        print(
            f"{name:>10}  noisy shape={payload['signal_noisy'].shape}  "
            f"noise_sigma={payload['noise_sigma']:.6f}"
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_shape": list(args.spatial_shape),
        "window": list(args.window),
        "seed": args.seed,
        "noise_fraction": args.noise_fraction,
        "num_spatial_dims": 3,
        "opt_shrink": args.opt_shrink,
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print()
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
