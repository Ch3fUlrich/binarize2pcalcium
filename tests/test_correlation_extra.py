import pytest
import numpy as np
from binarize2pcalcium.correlation import get_correlations, correlations_parallel, get_corr

def test_correlations_parallel_no_dff():
    rasters = np.random.randint(0, 2, (5, 300)).astype(float)
    rasters_dff = np.random.randn(5, 300) * 0.1
    result = correlations_parallel(
        [0, 1, 2], rasters, rasters_dff,
        binning_window=1, subsample=1, scale_by_DFF=True, zscore=False,
    )
    assert len(result) == 15

def test_get_corr_subsample():
    t1 = np.random.randn(500)
    t2 = t1 * 0.5 + np.random.randn(500) * 0.1
    # Check if there is a subsample implementation
    # It appears not based on earlier runs, so what else was missed in get_corr?
    # Missed lines: 126-129, 137-140 in correlation.py which correspond to get_corr2, let's test get_corr2.
