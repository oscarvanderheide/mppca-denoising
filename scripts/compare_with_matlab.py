"""Run Python denoising and compare it against MATLAB reference outputs."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from matlab_compare_common import load_json, load_mat_array, save_mat, write_json
from mppca_denoising import denoise_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug") / "matlab_compare",
        help="Directory containing manifest.json and MATLAB outputs",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | mps (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size passed to denoise_tensor",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help=(
            "Absolute tolerance for pass/fail (default: 1e-3). "
            "GPU vs MATLAB comparisons have ~1e-4 absolute differences "
            "from non-deterministic accumulation on different hardware; "
            "use 1e-6 only for CPU-vs-CPU comparisons."
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help=(
            "Relative tolerance for pass/fail (default: 0.05). "
            "P and SNR_gain have up to 4%% relative differences from "
            "borderline rank-estimation on GPU vs CPU."
        ),
    )
    parser.add_argument(
        "--atol-p",
        type=float,
        default=0.05,
        help=(
            "Absolute tolerance for P pass/fail (default: 0.05). "
            "P is an averaged integer rank divided by patch count; a difference "
            "of 1 in a borderline patch out of ~25 counts gives abs error ~0.04."
        ),
    )
    parser.add_argument(
        "--atol-snr",
        type=float,
        default=2.0,
        help=(
            "Absolute tolerance for SNR_gain pass/fail (default: 2.0). "
            "SNR_gain is highly sensitive to P when P is small; a rank rounding "
            "of 1/N_patches per mode can cause absolute SNR differences >1.5 at "
            "borderline low-signal voxels even though relative RMSE stays tiny."
        ),
    )
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_optional(path: Path, key: str) -> np.ndarray | None:
    try:
        return load_mat_array(path, key)
    except KeyError:
        return None


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    finite_mask = np.isfinite(reference) & np.isfinite(candidate)
    nan_mask_matches = np.array_equal(np.isnan(reference), np.isnan(candidate))
    inf_mask_matches = np.array_equal(np.isinf(reference), np.isinf(candidate))

    if finite_mask.any():
        ref_f = reference[finite_mask]
        cand_f = candidate[finite_mask]
        abs_diff = np.abs(cand_f - ref_f)
        rel_diff = abs_diff / (np.abs(ref_f) + 1e-12)
        rmse = float(np.sqrt(np.mean(np.square(abs_diff))))
        max_abs_error = float(abs_diff.max())
        mean_abs_error = float(abs_diff.mean())
        max_rel_error = float(rel_diff.max())
        mean_rel_error = float(rel_diff.mean())
        if ref_f.size > 1:
            corr = float(np.corrcoef(ref_f.ravel(), cand_f.ravel())[0, 1])
        else:
            corr = float("nan")
    else:
        rmse = 0.0
        max_abs_error = 0.0
        mean_abs_error = 0.0
        max_rel_error = 0.0
        mean_rel_error = 0.0
        corr = float("nan")

    allclose = bool(
        nan_mask_matches
        and inf_mask_matches
        and np.allclose(reference, candidate, atol=atol, rtol=rtol, equal_nan=True)
    )

    return {
        "shape_matches": reference.shape == candidate.shape,
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "nan_mask_matches": nan_mask_matches,
        "inf_mask_matches": inf_mask_matches,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_rel_error": max_rel_error,
        "mean_rel_error": mean_rel_error,
        "rmse": rmse,
        "pearson_correlation": corr,
        "allclose": allclose,
    }


def summarise_metric(name: str, metrics: dict[str, Any]) -> str:
    status = "PASS" if metrics["allclose"] else "FAIL"
    corr = metrics["pearson_correlation"]
    corr_text = f"{corr:.10f}" if not math.isnan(corr) else "nan"
    return (
        f"  {name:<9} {status}  "
        f"max_abs={metrics['max_abs_error']:.3e}  "
        f"rmse={metrics['rmse']:.3e}  corr={corr_text}"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found at {manifest_path}. "
            "Run generate_matlab_comparison_cases.py first."
        )

    manifest = load_json(manifest_path)
    device = get_device(args.device)
    window = manifest["window"]
    opt_shrink = bool(manifest.get("opt_shrink", True))

    print("=" * 72)
    print("Comparing Python output against MATLAB reference")
    print("=" * 72)
    print(f"Manifest   : {manifest_path}")
    print(f"Device     : {device}")
    print(f"Window     : {window}")
    print(f"Opt shrink : {opt_shrink}")
    print()

    report_cases: list[dict[str, Any]] = []

    for case in manifest["cases"]:
        case_name = case["name"]
        input_path = output_dir / case["input_mat"]
        matlab_path = output_dir / case["matlab_output_mat"]
        python_path = output_dir / case["python_output_mat"]
        python_runtime_path = output_dir / case["python_runtime_json"]

        if not matlab_path.exists():
            raise SystemExit(
                f"MATLAB result missing for {case_name}: {matlab_path}. "
                "Run scripts/run_matlab_reference.m or the shell wrapper first."
            )

        noisy = np.asarray(load_mat_array(input_path, "signal_noisy"), dtype=np.float64)
        matlab_denoised = np.asarray(load_mat_array(matlab_path, "denoised"))
        matlab_sigma2 = load_optional(matlab_path, "Sigma2")
        matlab_p = load_optional(matlab_path, "P")
        matlab_snr_gain = load_optional(matlab_path, "SNR_gain")

        print(f"{case_name}:")
        print(f"  noisy shape : {noisy.shape}")
        print(f"  matlab file : {matlab_path}")

        start = time.perf_counter()
        denoised, sigma2, p_est, snr_gain = denoise_tensor(
            noisy,
            window,
            device=device,
            opt_shrink=opt_shrink,
            batch_size=args.batch_size,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        runtime_seconds = time.perf_counter() - start

        denoised_np = denoised.detach().cpu().numpy()
        sigma2_np = sigma2.detach().cpu().numpy()
        p_np = p_est.detach().cpu().numpy()
        snr_gain_np = snr_gain.detach().cpu().numpy()

        save_mat(
            python_path,
            denoised=denoised_np,
            Sigma2=sigma2_np,
            P=p_np,
            SNR_gain=snr_gain_np,
            runtime_seconds=np.array(runtime_seconds, dtype=np.float64),
        )
        write_json(
            python_runtime_path,
            {
                "case": case_name,
                "device": device,
                "runtime_seconds": runtime_seconds,
                "opt_shrink": opt_shrink,
                "batch_size": args.batch_size,
            },
        )

        comparisons = {
            "denoised": compare_arrays(
                matlab_denoised, denoised_np, atol=args.atol, rtol=args.rtol
            )
        }
        if matlab_sigma2 is not None:
            comparisons["Sigma2"] = compare_arrays(
                np.asarray(matlab_sigma2),
                sigma2_np,
                atol=args.atol,
                rtol=args.rtol,
            )
        if matlab_p is not None:
            comparisons["P"] = compare_arrays(
                np.asarray(matlab_p),
                p_np,
                atol=args.atol_p,
                rtol=args.rtol,
            )
        if matlab_snr_gain is not None:
            comparisons["SNR_gain"] = compare_arrays(
                np.asarray(matlab_snr_gain),
                snr_gain_np,
                atol=args.atol_snr,
                rtol=args.rtol,
            )

        for metric_name, metric_values in comparisons.items():
            print(summarise_metric(metric_name, metric_values))

        case_pass = all(metric["allclose"] for metric in comparisons.values())
        print(f"  python runtime : {runtime_seconds:.4f}s")
        print(f"  overall        : {'PASS' if case_pass else 'FAIL'}")
        print()

        report_cases.append(
            {
                "name": case_name,
                "input_mat": case["input_mat"],
                "matlab_output_mat": case["matlab_output_mat"],
                "python_output_mat": case["python_output_mat"],
                "python_runtime_seconds": runtime_seconds,
                "comparisons": comparisons,
                "overall_pass": case_pass,
            }
        )

    summary = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "device": device,
        "window": window,
        "opt_shrink": opt_shrink,
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": report_cases,
        "overall_pass": all(case["overall_pass"] for case in report_cases),
    }
    report_path = output_dir / "reports" / "python_vs_matlab.json"
    write_json(report_path, summary)

    print(f"Wrote comparison report to {report_path}")
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
