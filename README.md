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
| `batch_size` | `4096` | Patches per kernel call (auto-raised to 262144 on CUDA) |


## Benchmark

`(224, 240, 192, 5)` complex, 5×5×5 window:

| Device | Time |
|--------|------|
| RTX A5000 | 3.75 s |
| CPU | ~100 s |


## Citation

Olesen et al., "Tensor denoising of multidimensional MRI data", *Magn Reson Med*, 2022.
doi:[10.1002/mrm.29478](https://doi.org/10.1002/mrm.29478)
