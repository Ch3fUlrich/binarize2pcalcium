"""Tests for correlation and deduplication functions."""
import numpy as np
import pytest
from binarize2pcalcium.binarize2pcalcium import (
    Calcium,
    get_correlations,
    get_corr2,
    make_correlation_array,
    correlations_parallel,
    signaltonoise,
)


class TestGetCorrelations:
    """Tests for get_correlations standalone function."""

    def test_get_correlations(self):
        """Test get_correlations with a mock Calcium object."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        n_cells = 5
        c.corr_array = np.zeros((n_cells, n_cells, 2), 'float32')
        c.corr_array[:, :, 0] = np.random.rand(n_cells, n_cells) * 0.5
        c.corr_array[:, :, 1] = np.random.rand(n_cells, n_cells) * 0.05

        ids = np.array([0, 1, 2, 3, 4])
        corrs = get_correlations(ids, c)

        assert len(corrs) == n_cells * (n_cells - 1) // 2


class TestMakeCorrelationArray:
    """Tests for make_correlation_array."""

    def test_make_correlation_array(self):
        """Test building correlation array from list of correlations."""
        corrs = [
            [0, 1, 0.85, 0.001],
            [0, 2, 0.42, 0.050],
            [1, 2, 0.31, 0.200],
        ]
        n_cells = 3
        result = make_correlation_array(corrs, n_cells)
        assert result.shape == (n_cells, n_cells, 2)
        assert result[0, 1, 0] == 0.85
        assert result[0, 1, 1] == 0.001
        assert result[1, 2, 0] == 0.31

    def test_make_correlation_array_empty(self):
        """Test with empty correlations list."""
        result = make_correlation_array([], 5)
        assert result.shape == (5, 5, 2)
        assert np.all(result == 0)


class TestCorrelationsParallel:
    """Tests for correlations_parallel."""

    def test_correlations_parallel(self):
        """Test correlations_parallel with small data."""
        rasters = np.random.randint(0, 2, (5, 300)).astype(float)
        rasters_dff = np.random.randn(5, 300) * 0.1
        ids = [0, 1, 2]

        result = correlations_parallel(
            ids, rasters, rasters_dff,
            binning_window=1,
            subsample=1,
            scale_by_DFF=False,
            zscore=False,
        )

        assert len(result) == len(ids) * rasters.shape[0]
        for row in result:
            assert len(row) == 4  # [k, p, corr, pval]

    def test_correlations_parallel_with_binning(self):
        """Test correlations_parallel with binning."""
        rasters = np.random.randint(0, 2, (3, 300)).astype(float)
        rasters_dff = np.random.randn(3, 300) * 0.1
        ids = [0, 1]

        result = correlations_parallel(
            ids, rasters, rasters_dff,
            binning_window=30,
            subsample=5,
            scale_by_DFF=True,
            zscore=True,
        )

        assert len(result) == len(ids) * rasters.shape[0]
