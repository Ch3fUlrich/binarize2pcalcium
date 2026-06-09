# Module Reference

Complete reference for every public function in the `binarize2pcalcium` package.
All modules live under `binarize2pcalcium/`.

---

## `filters.py` — Signal Processing Filters

Butterworth filter design and application, plus median filter.

### `butter_highpass(cutoff: float, fs: float, order: int = 5) → tuple`

Design a Butterworth highpass filter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff` | `float` | *(required)* | Cutoff frequency in Hz |
| `fs` | `float` | *(required)* | Sampling frequency in Hz |
| `order` | `int` | `5` | Filter order |

Returns `(b, a)` filter coefficients.

### `butter_highpass_filter(data: ndarray, cutoff: float, fs: float, order: int = 5) → ndarray`

Apply Butterworth highpass filter to 1D data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `ndarray` | *(required)* | 1D input array |
| `cutoff` | `float` | *(required)* | Cutoff in Hz |
| `fs` | `float` | *(required)* | Sampling rate in Hz |
| `order` | `int` | `5` | Filter order |

### `butter_lowpass(cutoff: float, fs: float, order: int = 5) → tuple`

Design a Butterworth lowpass filter.  Same signature as `butter_highpass`.

### `butter_lowpass_filter(data: ndarray, cutoff: float, fs: float, order: int = 5) → ndarray`

Apply Butterworth lowpass filter to 1D data.

### `butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) → ndarray`

Design a Butterworth bandpass filter (SOS form).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowcut` | `float` | *(required)* | Low cutoff in Hz |
| `highcut` | `float` | *(required)* | High cutoff in Hz |
| `fs` | `float` | *(required)* | Sampling rate in Hz |
| `order` | `int` | `5` | Filter order |

Returns SOS (second-order sections) coefficients.

### `butter_bandpass_filter(data: ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) → ndarray`

Apply Butterworth bandpass filter to 1D data.

### `medfilt(x: ndarray, k: int) → ndarray`

Apply length-`k` median filter to 1D array.  Boundaries extended by repeating endpoints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ndarray` | *(required)* | 1D input array |
| `k` | `int` | *(required)* | Kernel size (must be odd) |

Raises `AssertionError` if `k` is even or `x` is not 1D.

---

## `thresholds.py` — Threshold Computation

Gaussian-fit threshold determination and signal-to-noise ratio.

### `signaltonoise(a: ndarray, axis: int = 0, ddof: int = 0) → ndarray`

Compute signal-to-noise ratio: `mean / std`.  Returns 0 where std = 0.

### `find_threshold_by_gaussian_fit(F_filtered: ndarray, percentile_threshold: float, dff_min: float) → list[float]`

Per-cell threshold via Gaussian fit to mode-mirrored distribution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `F_filtered` | `ndarray` | *(required)* | `[n_cells, n_timepoints]` filtered fluorescence |
| `percentile_threshold` | `float` | *(required)* | Cumulative probability (e.g. 0.99999) |
| `dff_min` | `float` | *(required)* | Minimum dF/F floor |

Algorithm: values below the mode are mirrored, a Gaussian is fit to the pooled
distribution, and the threshold is set at the given cumulative probability.

### `find_threshold_by_gaussian_fit_parallel(ll: list, percentile_threshold: float, snr_min: float, maximum_sigma: float = 100) → float`

Single-cell threshold (parallel worker version).  Takes `[F_detrended, cell_id]` pair.
If fitted sigma exceeds `maximum_sigma`, threshold is set to 1.

---

## `preprocess.py` — Trace Preprocessing

Standardization, dF/F computation, detrending, filtering.

### `standardize(traces: ndarray) → ndarray`

Standardize per cell to [0, 1]: subtract median, divide by range.

### `compute_dff(F: ndarray, data_type: str = '2p') → ndarray`

Compute dF/F from raw fluorescence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `F` | `ndarray` | *(required)* | `[n_cells, n_timepoints]` raw fluorescence |
| `data_type` | `str` | `'2p'` | `'2p'` → (F−F0)/F0; `'1p'` → F−F0 |

### `low_pass_filter(traces: ndarray, high_cutoff: float, sample_rate: float, order: int = 1) → ndarray`

Apply lowpass filter to each cell's trace.  Default `order=1` matches the original
`Calcium` class logic (gentle filtering).

### `high_pass_filter(traces: ndarray, low_cutoff: float, sample_rate: float, order: int = 1) → ndarray`

Apply highpass filter to each cell's trace.

### `band_pass_filter(traces: ndarray, low_cutoff: float, high_cutoff: float, sample_rate: float, order: int = 1) → ndarray`

Apply bandpass (Chebyshev-style) filter to each cell's trace.

### `detrend_traces(traces: ndarray, sample_rate: float, detrend_model_order: int = 1, detrend_filter_threshold: float = 0.001, mode_window: int | None = None) → ndarray`

Detrend by fitting polynomial to very-lowpass filtered version and subtracting.
Also does mode-based baseline removal.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traces` | `ndarray` | *(required)* | `[n_cells, n_timepoints]` |
| `sample_rate` | `float` | *(required)* | Hz |
| `detrend_model_order` | `int` | `1` | Polynomial order for trend fit |
| `detrend_filter_threshold` | `float` | `0.001` | Lowpass cutoff for trend extraction |
| `mode_window` | `int \| None` | `None` | Window size for piecewise mode subtraction (None = global) |

### `filter_model(traces: ndarray, sample_rate: float, detrend_model_order: int = 1) → ndarray`

Alternative detrend method: fits line through median of first/last 10K points (order 1),
quadratic (order 2), or lowpass + polynomial (order > 2).

### `wavelet_filter(traces: ndarray, sample_rate: float) → ndarray`

Wavelet-based denoising using db3 decomposition.  Removes approximation coefficients
from the wavelet decomposition and subtracts the reconstructed detail.

---

## `binarization.py` — Core Binarization

Threshold-based binary event detection.

### `binarize_onphase(traces: ndarray, thresholds: list[float], min_width_event: int = 15) → ndarray`

Binarize traces: values ≥ threshold → 1, short events discarded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traces` | `ndarray` | *(required)* | `[n_cells, n_timepoints]` |
| `thresholds` | `list[float]` | *(required)* | Per-cell threshold values |
| `min_width_event` | `int` | `15` | Minimum event width in samples |

Uses `scipy.signal.find_peaks` + `peak_widths` to detect and filter events.

### `binarize_std(traces: ndarray, thresh: float = 2) → tuple[ndarray, ndarray]`

Binarize by std-based threshold: value ≥ `std * thresh` → event.

Returns `(binarized_traces, anti_aliased_traces)`.

### `binarize_derivative(traces: ndarray) → tuple[ndarray, ndarray]`

Compute gradient of traces (alternative event detection based on slope).

### `scale_binarized(traces: ndarray, traces_scale: ndarray, min_event_amplitude: float = 0) → ndarray`

Scale binarized event amplitudes by the DFF sum within each event window.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traces` | `ndarray` | *(required)* | Binarized array |
| `traces_scale` | `ndarray` | *(required)* | DFF-scaled array (same shape) |
| `min_event_amplitude` | `float` | `0` | Minimum amplitude for valid events |

### `binarize_upphase(F_filtered: ndarray, thresholds: list[float], min_width_event: int = 7, der_min_slope: float = 0, F_detrended: ndarray | None = None) → ndarray`

Binarize the rising phase.  Gradient is computed from `F_detrended` (defaults to
`F_filtered` if not provided) and portions with slope ≤ `der_min_slope` are zeroed
before `binarize_onphase` is called on the remaining signal.

### `smooth_traces(traces: ndarray, sample_rate: float) → ndarray`

Smooth traces with exponential convolution + lowpass filter.  Uses `scipy.signal.windows.exponential`.

---

## `correlation.py` — Correlation Computation

Pearson correlation functions for fluorescence traces and binarized rasters.

### `get_corr(temp1: ndarray, temp2: ndarray, zscore: bool = False, n_tests: int = 500) → list`

Compute Pearson correlation between two 1D arrays.  Returns `[corr, pval]` or `[corr, zscore]`.

### `get_corr2(temp1: ndarray, temp2: ndarray, zscore: bool, n_tests: int = 1000, min_number_bursts: int = 0) → tuple`

Pearson correlation with burst filtering and z-score via shuffling.
Returns `(corr_original, corr_z_array)`.

### `correlations_parallel(ids: list[int], rasters: ndarray, rasters_DFF: ndarray, binning_window: int = 30, subsample: int = 5, scale_by_DFF: bool = True, zscore: bool = False) → list`

Pairwise correlations for specified cell IDs with optional binning and DFF scaling.

### `make_correlation_array(corrs: list, n_cells: int) → ndarray`

Build 3D correlation array `[n_cells, n_cells, 2]` where `[..., 0]` = corr, `[..., 1]` = pval.

### `get_correlations(ids: ndarray, corr_array: ndarray) → ndarray`

Extract correlation values for specific cell IDs from a correlation array.

---

## `overlap.py` — Spatial Overlap Analysis

Footprint overlap detection and inter-cell distance computation.

### `array_row_intersection(a: ndarray, b: ndarray) → ndarray`

Find rows in `a` that also exist in `b`.

### `find_overlaps(ids: ndarray, footprints: ndarray) → list`

Find overlapping pixels between cell footprints.
Returns `[cell1, cell2, overlapping_pixels]` entries.

### `find_overlaps1(ids: ndarray, footprints: ndarray) → list`

Like `find_overlaps` but also returns `[percent_cell1, percent_cell2]`.

### `find_overlaps2(ids: ndarray, footprints: ndarray, footprints_bin: ndarray) → list`

Like `find_overlaps1` but with binarized footprint pre-check for speed.

### `make_overlap_database(res: list) → DataFrame`

Convert overlap results to a pandas DataFrame with columns:
`cell1, cell2, pixels_overlap, percent_cell1, percent_cell2`.

### `find_inter_cell_distance(footprints: ndarray) → tuple[ndarray, ndarray]`

Pairwise Euclidean distances between cell centres (median of footprint pixels).
Returns `(distance_matrix, upper_triangle_matrix)`.  Self-distance = NaN.

### `alpha_shape(points: ndarray, alpha: float = 0.6) → tuple`

Compute the alpha shape (concave hull) of a set of points using Delaunay triangulation.
Uses `shapely.ops.unary_union`.

---

## `dedup.py` — Cell Deduplication

Graph-based duplicate cell removal.

### `it_count(it) → tuple`

Count elements in an iterator without consuming subsequent elements.

### `del_highest_connected_nodes_without_corr(G: Graph) → tuple`

Remove highest-ID hub nodes from connected components, ignoring correlation values.

### `del_highest_connected_nodes(nn: set, c) → tuple`

Remove hub nodes from a correlated component.  Ties broken by lowest SNR.

### `del_lowest_snr(nn: set, c) → tuple`

Remove lowest-SNR cells from a correlated component.

---

## `pca.py` — Dimensionality Reduction

PCA, UMAP, TSNE, and longitudinal analysis functions.

### `run_UMAP(data: ndarray, n_neighbors: int = 50, min_dist: float = 0.1, n_components: int = 3, metric: str = 'euclidean') → ndarray`

Run UMAP dimensionality reduction.

### `compute_PCA(X: ndarray, data_dir: str, suffix1: str = '', suffix2: str = '', recompute: bool = True, save: bool = True) → tuple`

Run PCA with result caching to disk.  Returns `(pca_object, X_pca)`.

### `compute_TSNE(X: ndarray, data_dir: str, n_components: int = 2, perplexity: float = 100, learning_rate: float = 10) → ndarray`

Run TSNE with result caching.

### `compute_UMAP(X: ndarray, root_dir: str, n_components: int = 3, text: str = '') → ndarray`

Run UMAP with result caching.

### `fit_curves_aucs(aucs, fig1, ax1, ax2, fig3, ax3, animal_id, root_dir, binarization_method, clrs)`

Fit linear regression to AUC values for longitudinal analysis.  Modifies matplotlib axes in-place.

### `fit_curves_general(df, aucs, ax, animal_id, clr) → ax`

Fit linear regression for general AUC-vs-time analysis.  Returns the modified axes.

### `load_pca_animal(root_dir: str, animal_id: str, binarization_method: str, cell_randomization: bool, quiescent: bool) → tuple`

Load PCA AUC results for a specific animal.  Returns `(aucs, n_neurons)` or `(None, None)`.

---

## `io.py` — Data I/O

Loading Suite2p and Inscopix data formats.

### `load_suite2p(data_dir: str) → tuple`

Load Suite2p output: expects `stat.npy`, `F.npy`, `iscell.npy`, `spks.npy`, `ops.npy`.
Returns `(F, stat, iscell, spks, ops)`.

### `load_inscopix(data_dir: str) → ndarray`

Load Inscopix CNMF-E CSV data.  Returns `[n_cells, n_timepoints]` array.

---

## `data_simulation.py` — Synthetic Data

Realistic GCaMP-based calcium imaging simulation.

### `SimulationConfig` (dataclass)

Configuration for synthetic data generation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_cells` | `int` | `30` | Number of cells |
| `n_timepoints` | `int` | `9000` | Recording length in samples |
| `sample_rate` | `float` | `30.0` | Hz |
| `tau_rise_mean` | `float` | `0.08` | GCaMP rise time (s) |
| `tau_rise_std` | `float` | `0.02` | Rise time variability |
| `tau_decay_mean` | `float` | `0.6` | GCaMP decay time (s) |
| `tau_decay_std` | `float` | `0.15` | Decay time variability |
| `amplitude_mean` | `float` | `0.30` | Peak dF/F |
| `amplitude_std` | `float` | `0.10` | Amplitude variability |
| `event_rate` | `float` | `0.04` | Events / second / cell |
| `baseline_photons` | `float` | `300.0` | Mean baseline photon count |
| `baseline_std` | `float` | `60.0` | Cell-to-cell baseline variability |
| `read_noise_std` | `float` | `4.0` | Gaussian read noise (photons) |
| `drift_scale` | `float` | `0.02` | Slow baseline drift per frame |
| `seed` | `int` | `42` | Random seed |

### `simulate_calcium_data(config: SimulationConfig | None = None) → tuple[ndarray, ndarray]`

Generate realistic fluorescence traces.  Returns `(F_noisy, dff_true)`.

### `simulate_and_binarize(config, binarize_kwargs) → tuple`

Simulate data and run binarization in one call.
Returns `(result, event_gt, dff_true)` where `result` is a `BinarizationResult`.
