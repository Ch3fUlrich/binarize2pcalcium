# Configuration Reference

The YAML config files define all parameters for the binarization pipeline.
Values are extracted from the original `Calcium` class defaults.

## Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | All parameters with detailed comments (2P defaults) |
| `config/2p_pipeline.yaml` | Pre-configured for two-photon (Suite2p) data |
| `config/1p_pipeline.yaml` | Pre-configured for one-photon (Inscopix) data |

## Complete parameter reference

The default values shown come from `Calcium.set_default_parameters_2p()` (for 2p)
and `Calcium.set_default_parameters_1p()` (for 1p) in the original source code.

### Data Input

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `data_type` | `2p` | `1p` | `'2p'` = two-photon, `'1p'` = Inscopix / fibre photometry |
| `sample_rate` | `30` | `20` | Imaging sampling rate in Hz |

### Signal Filtering

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `high_cutoff` | `0.5` | `1.0` | Lowpass filter cutoff (Hz). Lower = smoother traces. Controls how much the Ca trace is smoothed before binarization — higher values preserve more temporal precision but produce more split events within a single Ca transient |
| `low_cutoff` | `0.005` | `0.005` | Highpass filter cutoff for bandpass Chebyshev filter step |

### Detrending (photobleaching removal)

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `detrend_model_order` | `1` | `1` | Polynomial order for trend fit. `1` = linear, `2` = quadratic |
| `detrend_model_type` | `mode` | `mode` | `'mode'` = mode-based baseline removal; `'polynomial'` = polynomial only |
| `detrend_filter_threshold` | `0.001` | `0.001` | Very-lowpass cutoff (Hz) to extract bleaching trend before polynomial fitting |
| `mode_window` | `900` | `600` | Sliding-window width (frames) for piecewise mode-based baseline detection. `null` = global mode over entire trace. Default = 30 s × sample_rate |

### Threshold Detection

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `percentile_threshold` | `0.99999` | `0.99999` | Gaussian-fit cumulative probability for threshold. The pipeline fits a Gaussian to the mode-centred lower half of the dF/F distribution (mode-mirroring method). The threshold is set at the dF/F value where the cumulative probability exceeds this percentile. `0.99999` = very conservative (only extreme bursts); `0.9999` = slightly more permissive; `0.999` = permissive (may detect noise events) |
| `dff_min` | `0.05` | `0.10` | Minimum dF/F floor for a valid event. Even if the Gaussian-fit threshold is lower, events must exceed this. 1P uses `0.10` because Inscopix data sometimes returns negative dF/F values from baseline subtraction |
| `maximum_std_of_signal` | `0.08` | `0.03` | If the standard deviation of a cell's filtered signal exceeds this, its threshold is set to 1 (effectively removed from binarization). 1P uses `0.03` because Inscopix data is noisier |

### Moment-Based Threshold Adjustment (primarily for 1P)

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `moment_flag` | `false` | `true` | Enable moment-based threshold adjustment. Checks the skewness-like moment of the Ca distribution; if it exceeds `moment_threshold`, the per-cell threshold is overridden with `moment_scaling`. Needed for Inscopix data which often has skewed distributions |
| `moment` | `2` | `2` | Moment order to compute (`scipy.stats.moment` order parameter) |
| `moment_threshold` | `0.01` | `0.01` | Moment value above which a cell is considered "bad" and gets the `moment_scaling` threshold |
| `moment_scaling` | `0.5` | `0.5` | Replacement dF/F threshold for cells that exceed `moment_threshold` |

### Event Detection

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `min_width_event_onphase` | `30` | `20` | Minimum event width in samples. Events narrower than this are discarded. 2P: 1 second at 30 Hz. 1P: 1 second at 20 Hz |
| `min_width_event_upphase` | `10` | `6` | Minimum upphase event width in samples. 2P: ~333 ms at 30 Hz. 1P: ~300 ms at 20 Hz |
| `der_min_slope` | `0` | `0` | Minimum derivative for upphase detection. Portions of the trace with slope ≤ this are zeroed before binarization. `0` = keep only rising portions |

### Processing Flags

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `use_upphase` | `true` | `true` | Compute both onphase AND upphase binarization |
| `remove_ends` | `false` | `false` | Replace first/last N samples with low-amplitude noise to remove filter edge artifacts. Default is `false` in both pipelines |
| `show_plots` | `true` | `true` | Display diagnostic plots during processing |
| `parallel_flag` | `true` | *(not in 1P config)* | Use parallel processing for threshold computation |

### Save Flags

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `save_python` | `true` | `true` | Save output as `.npz` file |
| `save_matlab` | `false` | `true` | Save output as `.mat` file. 1P defaults to `true` |

### Output

| Parameter | 2P default | 1P default | Description |
|-----------|-----------|-----------|-------------|
| `save_results` | `null` | `null` | Path to save results (`.npz`). `null` = skip saving. Example: `"binarized_results.npz"` |

## Using configs in code

```python
import yaml
from binarize2pcalcium import binarize

# Load pre-configured pipeline
with open('config/2p_pipeline.yaml') as f:
    params = yaml.safe_load(f)

# Pass to binarize (filter out null values)
result = binarize(F, **{k: v for k, v in params.items() if v is not None})
```

## Parameter tuning guide

### If you get too many events (false positives)
- Increase `percentile_threshold` (e.g. 0.99999 → 0.999999)
- Increase `dff_min` (e.g. 0.05 → 0.08)
- Decrease `high_cutoff` (e.g. 0.5 → 0.3) — smoother traces, fewer split events
- Increase `min_width_event_onphase` / `min_width_event_upphase`

### If you get too few events (false negatives)
- Decrease `percentile_threshold` (e.g. 0.99999 → 0.999)
- Decrease `dff_min` (e.g. 0.05 → 0.02)
- Increase `high_cutoff` (e.g. 0.5 → 1.0) — less smoothing, better temporal precision
- Decrease `min_width_event_onphase` / `min_width_event_upphase`

### If Inscopix data has weird noise distributions
- Set `moment_flag: true`
- Adjust `moment_threshold` — look at `moment_distribution.png` output
- Set `moment_scaling` to a reasonable dF/F floor (e.g. 0.3–0.5)
