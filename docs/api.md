# API Quick Reference

## Primary API

```python
from binarize2pcalcium import binarize, BinarizationResult

# Binarize a fluorescence matrix
result = binarize(F, sample_rate=30, data_type='2p')

# Access results
result.onphase       # ndarray[n_cells, n_timepoints]  — binarized onphase events
result.upphase       # ndarray[n_cells, n_timepoints]  — binarized upphase events
result.dff           # ndarray[n_cells, n_timepoints]  — dF/F traces
result.F_detrended   # ndarray[n_cells, n_timepoints]  — detrended+filtered
result.F_filtered    # ndarray[n_cells, n_timepoints]  — lowpass-filtered (pre-detrend)
result.thresholds    # list[float]                      — per-cell thresholds
result.sample_rate   # float                            — sampling rate used
result.data_type     # str                              — '2p' or '1p'

# Save / load
result.save('results.npz')
loaded = BinarizationResult.load('results.npz')
```

## Module imports

```python
# Filters
from binarize2pcalcium.filters import (
    butter_highpass, butter_highpass_filter,
    butter_lowpass, butter_lowpass_filter,
    butter_bandpass, butter_bandpass_filter,
    medfilt,
)

# Thresholds
from binarize2pcalcium.thresholds import (
    signaltonoise,
    find_threshold_by_gaussian_fit,
    find_threshold_by_gaussian_fit_parallel,
)

# Preprocessing
from binarize2pcalcium.preprocess import (
    standardize, compute_dff,
    low_pass_filter, high_pass_filter, band_pass_filter,
    detrend_traces, filter_model, wavelet_filter,
)

# Binarization
from binarize2pcalcium.binarization import (
    binarize_onphase, binarize_std, binarize_derivative,
    scale_binarized, binarize_upphase, smooth_traces,
)

# Correlation
from binarize2pcalcium.correlation import (
    get_corr, get_corr2, correlations_parallel,
    make_correlation_array, get_correlations,
)

# Overlap
from binarize2pcalcium.overlap import (
    array_row_intersection,
    find_overlaps, find_overlaps1, find_overlaps2,
    make_overlap_database, find_inter_cell_distance,
    alpha_shape,
)

# Deduplication
from binarize2pcalcium.dedup import (
    it_count,
    del_highest_connected_nodes,
    del_highest_connected_nodes_without_corr,
    del_lowest_snr,
)

# PCA / dimensionality reduction
from binarize2pcalcium.pca import (
    run_UMAP, compute_PCA, compute_TSNE, compute_UMAP,
    fit_curves_aucs, fit_curves_general, load_pca_animal,
)

# I/O
from binarize2pcalcium.io import load_suite2p, load_inscopix

# Simulation
from binarize2pcalcium.data_simulation import (
    SimulationConfig,
    simulate_calcium_data,
    simulate_and_binarize,
)
```

## Common patterns

### Binarize with custom thresholds

```python
result = binarize(
    F,
    sample_rate=30,
    high_cutoff=1.0,              # less smoothing
    percentile_threshold=0.999,   # more permissive
    dff_min=0.03,                 # lower floor
    min_width_onphase=10,         # shorter events accepted
)
```

### Binarize 1P data

```python
result = binarize(
    F,
    sample_rate=20,
    data_type='1p',
    high_cutoff=1.0,
    dff_min=0.10,
    min_width_onphase=20,
    min_width_upphase=6,
)
```

### Binarize from config file

```python
import yaml
with open('config/2p_pipeline.yaml') as f:
    params = yaml.safe_load(f)
result = binarize(F, **{k: v for k, v in params.items() if v is not None})
```

### Simulate and test

```python
from binarize2pcalcium.data_simulation import SimulationConfig, simulate_and_binarize

cfg = SimulationConfig(n_cells=30, n_timepoints=9000)
result, event_gt, dff_true = simulate_and_binarize(cfg)

# Compare detected vs ground truth
recall = (result.onphase & event_gt).sum() / event_gt.sum()
print(f"Recall: {recall:.2%}")
```

### Compute inter-cell correlations

```python
from binarize2pcalcium.correlation import correlations_parallel, make_correlation_array

corrs = correlations_parallel(
    ids=[0, 1, 2, 3, 4],
    rasters=result.onphase,
    rasters_DFF=result.dff,
)

corr_array = make_correlation_array(corrs, n_cells=result.onphase.shape[0])
print(corr_array[0, 1, 0])  # correlation between cell 0 and cell 1
```

### Find overlapping cell footprints

```python
from binarize2pcalcium.overlap import find_overlaps1, make_overlap_database

# footprints: [n_cells, height, width] array
overlaps = find_overlaps1(np.arange(n_cells), footprints)
df = make_overlap_database([overlaps])  # pandas DataFrame
print(df[df['percent_cell1'] > 0.5])   # cells with >50% overlap
```
