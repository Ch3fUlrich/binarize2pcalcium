# binarize2pcalcium

Binarization pipeline for converting continuous 2P and 1P calcium imaging traces
into binarized event time series. Now with a single-function API — no classes needed.

This code was originally written by [Catalin](https://github.com/catubc).

## Quick Start

```python
from binarize2pcalcium import binarize

# From a data matrix (shape: [n_cells, n_timepoints])
result = binarize(F, sample_rate=30, data_type='2p')

# From Suite2p output directory
result = binarize('/path/to/suite2p/plane0/', data_type='2p')

# Access results
result.onphase   # binarized event traces
result.upphase   # binarized rising-phase events
result.dff       # dF/F traces
result.thresholds  # per-cell thresholds
```

## Setup (Development with uv)

This project uses [uv](https://docs.astral.sh/uv/) as the Python package manager.

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Install for development

```bash
git clone https://github.com/Ch3fUlrich/binarize2pcalcium
git switch whatever
cd binarize2pcalcium

# Create virtual environment and install all dependencies including dev/test
uv sync --extra dev

# Verify installation
uv run python -c "from binarize2pcalcium import binarize; print('OK')"
```

### Running tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=binarize2pcalcium --cov-report=term
```

## Module Structure

```
binarize2pcalcium/
    __init__.py      # Public API
    pipeline.py      # Main binarize() function + BinarizationResult
    filters.py       # Butterworth filters (highpass, lowpass, bandpass)
    thresholds.py    # Gaussian-fit threshold computation
    preprocess.py    # Standardization, detrending, dF/F, wavelet filter
    binarization.py  # Core binarization (onphase, derivative, scaling)
    correlation.py   # Pearson correlation functions
    overlap.py       # Spatial footprint overlap analysis
    dedup.py         # Cell deduplication utilities
    pca.py           # PCA, UMAP, TSNE dimensionality reduction
    io.py            # Data loading (Suite2p, Inscopix)
```

## Configuration via YAML

Pre-built config files in `config/`:

```yaml
# config/default.yaml — all parameters documented
data_type: 2p
sample_rate: 30
high_cutoff: 2.0
detrend_order: 1
percentile_threshold: 0.9999
dff_min: 0.05
use_upphase: true
remove_ends: true
```

Or load and pass to `binarize()`:

```python
import yaml
with open('config/2p_pipeline.yaml') as f:
    params = yaml.safe_load(f)
result = binarize(F, **params)
```

## Documentation

Comprehensive reference documentation is available in the `docs/` folder:

- **`docs/index.md`** — landing page with links to all documentation files
- **`docs/pipeline.md`** — full reference for the `binarize()` function and `BinarizationResult`
- **`docs/modules.md`** — every public function across all modules
- **`docs/config.md`** — configuration YAML parameter reference
- **`docs/api.md`** — quick-start API cheatsheet
- **`docs/refactoring.md`** — overview of the monolithic-to-modular refactoring

## Jupyter Notebooks

Example notebooks are in the `notebooks/` directory (repo root):

- `2p_analysis.ipynb` — Full 2P binarization workflow with visualization
- `1p_analysis.ipynb` — 1P (Inscopix) binarization workflow

## Required .yaml files (for Suite2p data loading)

- `ANIMAL_ID.yaml`  ← inside the animal directory
- `SESSION_ID.yaml` ← inside the session directory

See example files provided in `binarize2pcalcium/`.

## Install via pip

```bash
pip install binca
```

## Legacy Usage

The original `Calcium` class still exists in `binarize2pcalcium/binarize2pcalcium.py`
for backward compatibility, but the recommended API is the new `binarize()` function.
