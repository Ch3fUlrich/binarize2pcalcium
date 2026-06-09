"""Tests for the pipeline module (the core binarize() function)."""
import numpy as np
import pytest
from binarize2pcalcium.pipeline import binarize, BinarizationResult


def test_binarize_basic():
    """Test the full pipeline with a simple data matrix."""
    F = np.random.randn(10, 500) * 0.1 + 0.5
    result = binarize(F, sample_rate=30, data_type='2p', verbose=False)
    assert isinstance(result, BinarizationResult)
    assert result.F_raw.shape == (10, 500)
    assert result.dff.shape == (10, 500)
    assert result.F_filtered.shape == (10, 500)
    assert result.F_detrended.shape == (10, 500)
    assert result.onphase.shape == (10, 500)
    assert result.upphase.shape == (10, 500)
    assert len(result.thresholds) == 10


def test_binarize_1p():
    """Test with 1P data type."""
    F = np.random.randn(5, 300) * 0.1 + 3.0
    result = binarize(F, sample_rate=20, data_type='1p', use_upphase=False)
    assert result.data_type == '1p'
    assert result.sample_rate == 20.0
    assert np.all(result.upphase == 0)  # upphase disabled
    assert result.onphase.shape == (5, 300)


def test_binarize_custom_thresholds():
    """Test with custom threshold parameters."""
    F = np.random.randn(8, 400) * 0.05 + 0.3
    result = binarize(
        F,
        sample_rate=30,
        percentile_threshold=0.99,
        dff_min=0.02,
    )
    for t in result.thresholds:
        assert t >= 0.02


def test_binarize_result_save_load(tmp_path):
    """Test save/load of BinarizationResult."""
    F = np.random.randn(5, 200) * 0.1 + 0.5
    result = binarize(F, sample_rate=30)
    path = tmp_path / "test_results.npz"
    result.save(str(path))
    loaded = BinarizationResult.load(str(path))
    assert loaded.sample_rate == 30.0
    assert np.array_equal(result.onphase, loaded.onphase)


def test_binarize_no_upphase():
    """Test pipeline with upphase disabled."""
    F = np.random.randn(5, 200) * 0.1 + 0.5
    result = binarize(F, use_upphase=False)
    assert np.all(result.upphase == 0)


def test_binarize_verbose():
    """Test pipeline with verbose output."""
    F = np.random.randn(3, 100) * 0.1 + 0.3
    result = binarize(F, verbose=True)
    assert result.onphase.shape == (3, 100)
