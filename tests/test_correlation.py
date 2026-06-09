"""Tests for correlation module."""
import numpy as np
import pytest
from binarize2pcalcium.correlation import (
    get_corr,
    get_corr2,
    correlations_parallel,
    make_correlation_array,
    get_correlations,
)


def test_get_corr():
    t1 = np.array([1, 2, 3, 4, 5])
    t2 = np.array([2, 4, 6, 8, 10])
    corr = get_corr(t1, t2)
    assert corr[0] > 0.99


def test_get_corr_constant():
    t1 = np.array([1, 1, 1, 1, 1])
    t2 = np.array([2, 4, 6, 8, 10])
    corr = get_corr(t1, t2)
    assert np.isnan(corr[0])


def test_get_corr_zscore():
    t1 = np.random.randn(500)
    t2 = t1 * 0.5 + np.random.randn(500) * 0.1
    corr = get_corr(t1, t2, zscore=True, n_tests=100)
    assert len(corr) == 2


def test_get_corr2():
    t1 = np.random.randn(500) * 0.1 + 0.5
    t2 = t1 * 0.8 + np.random.randn(500) * 0.05
    corr_orig, corr_z = get_corr2(t1, t2, zscore=True, n_tests=50)
    assert corr_orig[0] > 0


def test_get_corr2_constant():
    t1 = np.ones(100)
    t2 = np.random.randn(100)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False)
    assert np.isnan(corr_orig[0])


def test_correlations_parallel():
    rasters = np.random.randint(0, 2, (5, 300)).astype(float)
    rasters_dff = np.random.randn(5, 300) * 0.1
    result = correlations_parallel(
        [0, 1, 2], rasters, rasters_dff,
        binning_window=1, subsample=1, scale_by_DFF=False, zscore=False,
    )
    assert len(result) == 15  # 3 ids * 5 cells


def test_make_correlation_array():
    corrs = [[0, 1, 0.85, 0.001], [0, 2, 0.42, 0.050], [1, 2, 0.31, 0.200]]
    result = make_correlation_array(corrs, 3)
    assert result.shape == (3, 3, 2)
    assert result[0, 1, 0] == 0.85


def test_get_correlations():
    corr_array = np.zeros((5, 5, 2), 'float32')
    corr_array[:, :, 0] = np.random.rand(5, 5) * 0.5
    ids = np.array([0, 1, 2, 3, 4])
    corrs = get_correlations(ids, corr_array)
    assert len(corrs) == 10  # 5*4/2


def test_get_corr2_no_zscore():
    t1 = np.random.randn(200)
    t2 = np.random.randn(200)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False)
    assert len(corr_orig) == 2


def test_get_corr2_min_bursts():
    t1 = np.zeros(500)
    t1[50] = 1
    t1[60] = 1
    t2 = np.random.randn(500)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False, min_number_bursts=3)
    assert np.isnan(corr_orig[0])
