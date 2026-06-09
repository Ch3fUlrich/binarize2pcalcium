# binarize2pcalcium

Binarization pipelines for converting continuous 2P calcium imaging traces into binarized time series.

This code was written by [Catalin](https://github.com/catubc).

## Setup (Development with uv)

This project uses [uv](https://docs.astral.sh/uv/) as the Python package manager.

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Install for development

```bash
# Clone the repository
git clone <repo-url>
cd binarize2pcalcium

# Create virtual environment and install all dependencies including dev/test
uv sync --extra dev

# Verify installation
uv run python -c "from binarize2pcalcium.binarize2pcalcium import Calcium; print('OK')"
```

### Running tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=binarize2pcalcium --cov-report=term

# Run specific test file
uv run pytest tests/test_filters_and_utils.py -v
```

### Test coverage notes

The test suite contains 85 tests across 5 test files covering:

- **Standalone functions**: Butterworth filters, signal-to-noise, correlation functions, array operations, iterator utilities
- **Calcium class methods**: Initialization, standardization, filtering (high/low/bandpass, median, wavelet, Chebyshev), binarization methods, detrending, smoothing, shuffle rasters, parameter defaults, SNR computation, moment-based thresholding
- **Correlation functions**: `get_correlations`, `make_correlation_array`, `correlations_parallel`
- **Overlap/spatial functions**: `find_overlaps`, `find_overlaps1`, `find_overlaps2`, `make_overlap_database`, `find_inter_cell_distance`

Functions that require real 2P imaging data files or produce matplotlib plots (e.g., `load_suite2p`, `load_inscopix`, `binarize_data`, `run_binarize`, `show_rasters`, `plot_traces`) are not covered by unit tests as they depend on external data and display backends.

## Required .yaml files

You will need 2 .yaml files to read the meta data of your recordings:

- `ANIMAL_ID.yaml`  <- inside the animal directory
- `SESSION_ID.yaml` <- inside the session directory

Please see example .yaml files provided in `binarize2pcalcium/`.

## Basic Usage

```python
from binarize2pcalcium import binarize2pcalcium as binca

data_dir = '/media/cat/2pdata'
animal_id = 'DON-011733'
session = '20230203'

c = binca.Calcium(data_dir, animal_id)

c.session = session
c.session_name = session

c.data_type = '2p'
c.remove_bad_cells = False
c.verbose = False                          # outputs additional information during processing
c.recompute_binarization = True           # recomputes binarization and other processing steps; False: loads from previous saved locations

# set flags to save matlab and python data
c.save_python = True         # save output as .npz file
c.save_matlab = False         # save output as .mat file

# manual thresholds for spike detection
c.dff_min = 0.05                  # min %DFF for [ca] burst to considered a spike (default 5%) overwrites percentile threshold parameter
c.percentile_threshold = 0.9999   # this is pretty fixed, we don't change it; we want [ca] bursts that are well outside the "physics-caused" noise
c.maximum_std_of_signal = 0.08     # if std of signal is greater than this, then we have a noisy signal and we don't want to binarize it
                                   #    - this is a very important flag!

#
c.binarize_data()
```

## Install via pip

```bash
pip install binca
```
