"""Tests for preprocess module."""
import numpy as np
import pytest
from binarize2pcalcium.preprocess import (
    standardize,
    compute_dff,
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    detrend_traces,
    filter_model,
    wavelet_filter,
)


def test_standardize():
    traces = np.random.randn(10, 100)
    std = standardize(traces)
    assert std.shape == traces.shape


def test_standardize_range():
    traces = np.random.randn(5, 200) * 2 + 3
    std = standardize(traces)
    for k in range(traces.shape[0]):
        assert np.isclose(np.max(std[k]) - np.min(std[k]), 1.0, atol=0.01)


def test_compute_dff_2p():
    F = np.random.randn(10, 100) * 0.2 + 10
    dff = compute_dff(F, data_type='2p')
    assert dff.shape == F.shape


def test_compute_dff_1p():
    F = np.random.randn(5, 100) * 0.2 + 5
    dff = compute_dff(F, data_type='1p')
    assert dff.shape == F.shape


def test_low_pass_filter():
    traces = np.random.randn(3, 100)
    filtered = low_pass_filter(traces, high_cutoff=2.0, sample_rate=30)
    assert filtered.shape == traces.shape


def test_high_pass_filter():
    traces = np.random.randn(3, 100)
    filtered = high_pass_filter(traces, low_cutoff=0.01, sample_rate=30)
    assert filtered.shape == traces.shape


def test_band_pass_filter():
    traces = np.random.randn(3, 100)
    filtered = band_pass_filter(traces, low_cutoff=0.5, high_cutoff=2.0, sample_rate=30)
    assert filtered.shape == traces.shape


def test_detrend_traces():
    traces = np.random.randn(5, 200) * 0.1 + np.linspace(0, 1, 200)
    result = detrend_traces(traces, sample_rate=30, detrend_model_order=1)
    assert result.shape == traces.shape


def test_filter_model_order1():
    traces = np.random.randn(3, 200) * 0.1
    result = filter_model(traces, sample_rate=30, detrend_model_order=1)
    assert result.shape == traces.shape


def test_filter_model_order2():
    traces = np.random.randn(3, 200) * 0.1
    result = filter_model(traces, sample_rate=30, detrend_model_order=2)
    assert result.shape == traces.shape


def test_wavelet_filter():
    traces = np.random.randn(1, 100)
    filtered = wavelet_filter(traces, sample_rate=30)
    assert filtered.shape == traces.shape
