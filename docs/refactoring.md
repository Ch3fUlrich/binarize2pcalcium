# Refactoring Overview

## From monolithic class to modular functions

The original `binarize2pcalcium/binarize2pcalcium.py` was a **single 4,111-line file**
containing a `Calcium` class with ~40+ methods alongside 38 standalone functions,
all in one module.  The refactoring (June 2026) decomposed this into **10 focused
modules** and replaced the class with a **single `binarize()` function**.

### Motivations

| Problem | Solution |
|---------|----------|
| 4,111-line file — impossible to navigate | 10 modules, each ≤ 330 lines |
| `Calcium` class required ~15 attributes set before use | `binarize(F, sample_rate=30, ...)` — one call |
| Class state made debugging painful | Pure functions with explicit parameters |
| No separation of concerns | Each module handles one domain |
| No pip-installable package structure | `pyproject.toml` with `uv` support |

### Before → After

```
BEFORE (monolithic):               AFTER (modular):
════════════════════                ═══════════════════
binarize2pcalcium/                  binarize2pcalcium/
  binarize2pcalcium.py  (4111 L)     __init__.py          Public API
                                     pipeline.py          binarize() entry point
                                     filters.py           Butterworth filters
                                     thresholds.py        Gaussian-fit thresholds
                                     preprocess.py        dF/F, detrend, wavelet
                                     binarization.py      Onphase/upphase detection
                                     correlation.py       Pearson correlations
                                     overlap.py           Spatial footprint analysis
                                     dedup.py             Cell deduplication
                                     pca.py               PCA, UMAP, TSNE
                                     io.py                Suite2p / Inscopix loading
                                     data_simulation.py   Synthetic data generation

config/                           config/
  default.yaml                      default.yaml
  1p_pipeline.yaml                  1p_pipeline.yaml
  2p_pipeline.yaml                  2p_pipeline.yaml

notebooks/                        notebooks/
  (inside package)                  (repo root)
  1p_analysis.ipynb                 1p_analysis.ipynb
  2p_analysis.ipynb                 2p_analysis.ipynb

tests/                            tests/
  5 test files                      7 test files (54 tests)
```

### Legacy compatibility

The original `binarize2pcalcium/binarize2pcalcium.py` is preserved at `legacy/binarize2pcalcium.py`
for backward compatibility.  You can still import the `Calcium` class:

```python
from binarize2pcalcium.binarize2pcalcium import Calcium  # legacy
```

But the recommended API is the new function-based approach:

```python
from binarize2pcalcium import binarize  # new
```

### Key API differences

| Old (`Calcium` class)                         | New (`binarize()` function)                    |
|-----------------------------------------------|------------------------------------------------|
| `c = Calcium(root_dir, animal_id, ...)`       | `result = binarize(F, sample_rate=30, ...)`    |
| `c.session_name = '...'`                      | (no session management — just pass the matrix) |
| `c.data_type = '2p'`                          | `data_type='2p'` parameter                     |
| `c.high_cutoff = 0.5`                         | `high_cutoff=0.5` parameter                    |
| `c.dff_min = 0.05`                            | `dff_min=0.05` parameter                       |
| `c.binarize_data()` → sets attributes on `c`  | Returns `BinarizationResult` dataclass         |
| Access via `c.F_onphase_bin`                  | `result.onphase`                               |

### Bug fixes during refactoring

The following bugs in the original code were fixed:

1. **`type=='linear'`** (legacy line 768) — comparison instead of assignment in `detrend()`.  Fixed by using `scipy.signal.detrend` correctly.

2. **Missing imports**: `umap`, `TSNE` (sklearn.manifold), `PCA` (sklearn.decomposition), `pickle` — all added as local imports in `pca.py`.

3. **`scipy.signal.exponential`** → `scipy.signal.windows.exponential` (deprecated in scipy ≥ 1.8).

4. **`shapely.ops.cascaded_union`** → `shapely.ops.unary_union` (removed in shapely ≥ 2.0).

5. **`make_overlap_database`** — missing `import pandas as pd` added.

6. **`correlations_parallel`** — unpacked 3 values from `get_corr2` which returns 2; fixed to `corr, corr_z = get_corr2(...)`.

7. **Pipeline data flow** — upphase binarization now correctly uses `F_filtered_saved` (pre-detrend) with gradient computed from `F_detrended`, matching the original logic exactly.
