#!/usr/bin/env bash
 
set -euo pipefail
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PYTHON_REPO}/debug/matlab_compare"
OUTPUT_DIR_IS_DEFAULT=1
MATLAB_TOOLBOX="${MATLAB_TOOLBOX:-/home/oheide/Documents/MATLAB/Tensor-MP-PCA}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
MATLAB_MODE="${MATLAB_MODE:-batch}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SKIP_GENERATE=0
SKIP_MATLAB=0
SKIP_COMPARE=0
 
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_RUNNER=("${PYTHON_BIN}")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_RUNNER=(python3)
else
  PYTHON_RUNNER=(python)
fi
 
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
 
Options:
  --output-dir PATH        Output directory for manifest and results
  --python-repo PATH       Repository root containing the scripts directory
  --matlab-toolbox PATH    Tensor-MP-PCA toolbox root
  --matlab-bin CMD         MATLAB executable (default: matlab)
  --matlab-mode MODE       batch | headless (default: batch, requires R2019a+)
  --python-bin PATH        Python executable to use instead of auto-detection
  --device NAME            Device for the Python comparison step
  --batch-size N           Batch size for the Python comparison step
  --skip-generate          Reuse an existing manifest and input .mat files
  --skip-matlab            Skip the MATLAB reference step
  --skip-compare           Skip the Python comparison step
  --help                   Show this help text
EOF
}
 
matlab_escape() {
  printf "%s" "$1" | sed "s/'/''/g"
}
 
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      OUTPUT_DIR_IS_DEFAULT=0
      shift 2
      ;;
    --python-repo)
      PYTHON_REPO="$2"
      if [[ ${OUTPUT_DIR_IS_DEFAULT} -eq 1 ]]; then
        OUTPUT_DIR="${PYTHON_REPO}/debug/matlab_compare"
      fi
      shift 2
      ;;
    --matlab-toolbox)
      MATLAB_TOOLBOX="$2"
      shift 2
      ;;
    --matlab-bin)
      MATLAB_BIN="$2"
      shift 2
      ;;
    --matlab-mode)
      MATLAB_MODE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_RUNNER=("$2")
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --skip-matlab)
      SKIP_MATLAB=1
      shift
      ;;
    --skip-compare)
      SKIP_COMPARE=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done
 
cd "${PYTHON_REPO}"
export PYTHONPATH="${PYTHON_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
 
echo "========================================================================"
echo "MATLAB/Python Tensor-MP-PCA comparison"
echo "========================================================================"
echo "Python repo    : ${PYTHON_REPO}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "MATLAB toolbox : ${MATLAB_TOOLBOX}"
echo "Python runner  : ${PYTHON_RUNNER[*]}"
echo "MATLAB bin     : ${MATLAB_BIN}"
echo "MATLAB mode    : ${MATLAB_MODE}"
echo
 
if [[ ${SKIP_GENERATE} -eq 0 ]]; then
  echo "[1/3] Generating deterministic phantom inputs"
  "${PYTHON_RUNNER[@]}" scripts/generate_matlab_comparison_cases.py --output-dir "${OUTPUT_DIR}"
  echo
fi
 
if [[ ${SKIP_MATLAB} -eq 0 ]]; then
  echo "[2/3] Running MATLAB reference"
  MATLAB_REPO_ESCAPED="$(matlab_escape "${PYTHON_REPO}/scripts")"
  MATLAB_OUTPUT_ESCAPED="$(matlab_escape "${OUTPUT_DIR}")"
  MATLAB_TOOLBOX_ESCAPED="$(matlab_escape "${MATLAB_TOOLBOX}")"
  MATLAB_RUN_CMD="addpath('${MATLAB_REPO_ESCAPED}'); run_matlab_reference('${MATLAB_OUTPUT_ESCAPED}', '${MATLAB_TOOLBOX_ESCAPED}');"
  if [[ "${MATLAB_MODE}" == "batch" ]]; then
    # -batch (R2019a+): truly headless, auto-exits, non-zero exit code on error.
    # No need to strip DISPLAY; -batch never initialises the GUI.
    "${MATLAB_BIN}" -batch "${MATLAB_RUN_CMD}"
  elif [[ "${MATLAB_MODE}" == "headless" ]]; then
    # Legacy -r mode for older MATLAB. Strip DISPLAY so no X window is opened.
    MATLAB_HEADLESS_CMD="try; ${MATLAB_RUN_CMD} exit(0); catch ME; disp(getReport(ME, 'extended')); exit(1); end;"
    env -u DISPLAY "${MATLAB_BIN}" -nodisplay -nosplash -nodesktop -r "${MATLAB_HEADLESS_CMD}"
  else
    echo "Unsupported --matlab-mode value: ${MATLAB_MODE}" >&2
    exit 1
  fi
  echo
fi

if [[ ${SKIP_COMPARE} -eq 0 ]]; then
  echo "[3/3] Running Python comparison"
  COMPARE_CMD=("${PYTHON_RUNNER[@]}" scripts/compare_with_matlab.py --output-dir "${OUTPUT_DIR}" --batch-size "${BATCH_SIZE}")
  if [[ -n "${DEVICE}" ]]; then
    COMPARE_CMD+=(--device "${DEVICE}")
  fi
  "${COMPARE_CMD[@]}"
  echo
fi
 
echo "Comparison workflow completed."