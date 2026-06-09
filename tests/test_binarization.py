"""Tests for binarization module."""
import numpy as np
import pytest
from binarize2pcalcium.binarization import (
    binarize_onphase,
    binarize_std,
    binarize_derivative,
    scale_binarized,
    binarize_upphase,
    smooth_traces,
)


def test_binarize_onphase():
    thresholds = [0.1, 0.1, 0.1]
    traces = np.random.randn(3, 200) * 0.3
    result = binarize_onphase(traces, thresholds, min_width_event=5)
    assert result.shape == traces.shape


def test_binarize_std():
    traces = np.random.randn(3, 100) * 0.5
    out, aa = binarize_std(traces, thresh=1.0)
    assert out.shape == traces.shape


def test_binarize_derivative():
    traces = np.random.randn(3, 100) * 0.5
    out, aa = binarize_derivative(traces)
    assert out.shape == traces.shape


def test_scale_binarized():
    traces = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    traces_scale = np.array([[0.0, 0.0, 2.0, 5.0, 2.0, 0.5, 0.0, 0.0]])
    scaled = scale_binarized(traces, traces_scale, min_event_amplitude=0)
    assert scaled.shape == traces.shape


def test_binarize_upphase():
    thresholds = [0.1, 0.1, 0.1]
    traces = np.random.randn(3, 200) * 0.3
    result = binarize_upphase(traces, thresholds, min_width_event=5)
    assert result.shape == traces.shape


def test_smooth_traces():
    traces = np.random.randn(3, 200) * 0.1
    result = smooth_traces(traces, sample_rate=30)
    assert result.shape == traces.shape
