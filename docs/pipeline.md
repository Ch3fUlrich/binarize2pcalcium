# Pipeline: The `binarize()` Function

The single entry point that replaces the entire `Calcium` class.  Pass in raw
fluorescence data (or a path) and get back binarized event traces.

## Quick start

```python
from binarize2pcalcium import binarize

# From a NumPy array
result = binarize(F, sample_rate=30, data_type='2p')

# From Suite2p output
result = binarize('/path/to/suite2p/plane0/', data_type='2p')

# Access results
result.onphase      # (n_cells, n_timepoints) binary onphase events
result.upphase      # (n_cells, n_timepoints) binary upphase events
result.dff          # dF/F traces
result.F_detrended  # detrended and filtered traces
result.thresholds   # list[float] per-cell detection thresholds
```

## Signature

```python
def binarize(
    F: np.ndarray | str,
    sample_rate: float = 30.0,
    data_type: str = '2p',
    high_cutoff: float = 0.5,
    detrend_order: int = 1,
    percentile_threshold: float = 0.99999,
    dff_min: float = 0.05,
    min_width_onphase: int = 30,
    min_width_upphase: int = 10,
    use_upphase: bool = True,
    remove_ends: bool = False,
    verbose: bool = False,
    **kwargs,
) -> BinarizationResult
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `F` | `ndarray \| str` | *(required)* | Raw fluorescence `[n_cells, n_timepoints]` or path to Suite2p `plane0/` / Inscopix CSV directory |
| `sample_rate` | `float` | `30.0` | Imaging sampling rate in Hz. 2P typical: 30; 1P typical: 20 |
| `data_type` | `str` | `'2p'` | `'2p'` = two-photon (dF/F = (F−F0)/F0); `'1p'` = one-photon (F−F0) |
| `high_cutoff` | `float` | `0.5` | Lowpass filter cutoff in Hz. Lower = smoother traces. 2P default from source: 0.5; 1P: 1.0 |
| `detrend_order` | `int` | `1` | Polynomial order for photobleaching/trend removal. 1 = linear |
| `percentile_threshold` | `float` | `0.99999` | Gaussian-fit cumulative probability for event threshold. Higher = more conservative |
| `dff_min` | `float` | `0.05` | Minimum dF/F for a valid event. 2P: 0.05; 1P: 0.10 |
| `min_width_onphase` | `int` | `30` | Minimum event width in samples. Events narrower than this are discarded |
| `min_width_upphase` | `int` | `10` | Minimum upphase event width in samples |
| `use_upphase` | `bool` | `True` | Whether to compute upphase binarization |
| `remove_ends` | `bool` | `False` | Replace first/last 300 samples with noise to remove filter edge artifacts |
| `verbose` | `bool` | `False` | Print progress information during pipeline execution |

## Pipeline Steps

The function executes these steps in order:

```
 Step 1: Load data (if path given)
    │
 Step 2: Compute dF/F
    │   ├─ 2p:  dF/F = (F − F0) / F0
    │   └─ 1p:  dF/F = F − F0
    │
 Step 3: Lowpass filter  (order=1 Butterworth, cutoff=high_cutoff)
    │
 Step 4: Remove edge artifacts  (optional, if remove_ends=True)
    │
 Step 5: Detrend  (polynomial fit subtraction)
    │
 Step 6: Compute per-cell thresholds  (Gaussian fit to mode-mirrored distribution)
    │
 Step 7: Binarize onphase  (threshold crossings → find_peaks → peak_width filter)
    │
 Step 8: Binarize upphase  (gradient from F_detrended, zero negative slope,
    │                       threshold crossings on F_filtered_saved)
    │
    └── Return BinarizationResult
```

### Important data flow detail

- **Onphase** uses `F_detrended` (detrended + filtered)
- **Upphase** uses `F_filtered_saved` (pre-detrend filtered copy) for binarization but computes the gradient from `F_detrended` — this matches the original `Calcium` class logic exactly

## BinarizationResult

A `dataclass` containing all pipeline outputs:

```python
@dataclass
class BinarizationResult:
    F_raw: np.ndarray          # Raw input [n_cells, n_timepoints]
    dff: np.ndarray            # dF/F traces [n_cells, n_timepoints]
    F_filtered: np.ndarray     # Lowpass-filtered dF/F (pre-detrend)
    F_detrended: np.ndarray    # Detrended + filtered traces
    onphase: np.ndarray        # Binary onphase events [n_cells, n_timepoints]
    upphase: np.ndarray        # Binary upphase events [n_cells, n_timepoints]
    thresholds: list[float]    # Per-cell threshold values
    sample_rate: float         # Sampling rate used
    data_type: str             # '2p' or '1p'
```

### Methods

| Method | Description |
|--------|-------------|
| `result.save(filepath)` | Save all results to a `.npz` file |
| `BinarizationResult.load(filepath)` | Class method — load results from `.npz` |

## Usage with config files

```python
import yaml
from binarize2pcalcium import binarize

# Load parameters from YAML
with open('config/2p_pipeline.yaml') as f:
    params = yaml.safe_load(f)

# Filter out null values and pass to binarize
result = binarize(F, **{k: v for k, v in params.items() if v is not None})
```

## Usage with simulated data

```python
from binarize2pcalcium.data_simulation import SimulationConfig, simulate_and_binarize

cfg = SimulationConfig(n_cells=30, n_timepoints=9000, sample_rate=30.0)
result, event_gt, dff_true = simulate_and_binarize(cfg)
# result: BinarizationResult with detected events
# event_gt: ground-truth binary events (uint8)
# dff_true: noise-free dF/F
```
