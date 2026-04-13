# mppca-denoising

MP-PCA denoising for MRI data ([Olesen et al., MRM 2022](https://doi.org/10.1002/mrm.29478)). PyTorch port of https://github.com/Neurophysics-CFIN/Tensor-MP-PCA. Not (yet) registered in PyPI.

## Install

```bash
uv add git+https://github.com/oscarvanderheide/mppca-denoising
```


## Usage

```python
from mppca_denoising import denoise_tensor

denoised, sigma2, P, snr_gain = denoise_tensor(data, window=[5, 5, 5], device="cuda")
```

`data` is a `(*spatial_dims, M)` NumPy array or PyTorch tensor, real or complex.

CLI usage:

```bash
uv run denoise_tensor input.npy --window 5 5 5
```

This writes `input_denoised.npz` containing `denoised`, `sigma2`, `p`, and `snr_gain`.
Use `--output result.mat` to write MATLAB output instead, and `--input-key` for `.npz` or `.mat`
inputs with multiple arrays.
For multidimensional MRI data, pass one window value per spatial dimension, e.g. `--window 5 5 5`.
If the CUDA eig fallback path is hit, the CLI prints fallback counts after the run.
The default `--solver auto` probes one batch and switches to `svd` when early `eigh` fallback activity indicates a pathological complex dataset.
You can still force `--solver eigh` or `--solver svd` manually.

### Solver Selection

`eigh` is the preferred fast path for normal cases, especially the synthetic benchmark in this repository.
`svd` is more robust for some real complex datasets whose small Gram matrices trigger many CUDA `eigh`
fallbacks.

`--solver auto` exists to choose between those modes cheaply:

1. It runs an `eigh` probe on one patch batch.
2. If that probe shows essentially no fallback activity, it keeps `eigh` for the full run.
3. If that probe already shows repeated `eigh` split or leaf fallbacks, it switches the full run to `svd`.

In practice, this means synthetic or well-behaved inputs stay on the faster `eigh` path, while
pathological complex inputs can automatically move to `svd` without forcing the slower backend for
every workload.


## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | — | Sliding window size per spatial dim |
| `mask` | `None` | Boolean mask; patches with no masked voxels are skipped |
| `stride` | `1` | Step size per spatial dim |
| `center_assign` | `False` | Accumulate only the centre voxel of each patch |
| `opt_shrink` | `True` | Frobenius-optimal singular-value shrinkage |
| `sigma2` | `None` | Known noise variance; estimated from data if `None` |
| `device` | auto | PyTorch device (`"cuda"`, `"cpu"`) |
| `dtype` | auto | `float32`/`complex64` on CUDA, else matches input |
| `batch_size` | `8192` | Patches processed per denoising batch; lower it if CUDA runs out of memory |
| `solver` | `auto` | Gram-matrix backend: `auto` probes one batch, `eigh` keeps the fast path, `svd` helps fallback-heavy complex data |


## Benchmark

Synthetic Shepp-Logan benchmark from `scripts/benchmark.py`, shape `(224, 240, 192, 5)`, 5×5×5 window:

| Device | Time |
|--------|------|
| RTX A5000 | 3.75 s |
| CPU | ~100 s |


## Citation

Olesen et al., "Tensor denoising of multidimensional MRI data", *Magn Reson Med*, 2022.
doi:[10.1002/mrm.29478](https://doi.org/10.1002/mrm.29478)
