"""Tests for thresholds module."""
import numpy as np
import pytest
from binarize2pcalcium.thresholds import (
    signaltonoise,
    find_threshold_by_gaussian_fit,
    find_threshold_by_gaussian_fit_parallel,
)


def test_signaltonoise():
    a = np.array([1, 2, 3, 4, 5])
    snr = signaltonoise(a)
    assert snr > 0
    b = np.array([5, 5, 5, 5, 5])
    assert signaltonoise(b) == 0


def test_signaltonoise_2d():
    a = np.random.randn(5, 100)
    snr = signaltonoise(a, axis=1)
    assert snr.shape == (5,)


def test_find_threshold_by_gaussian_fit():
    F_filtered = np.random.randn(2, 500) * 0.1
    thresholds = find_threshold_by_gaussian_fit(F_filtered, 0.99, 0.05)
    assert len(thresholds) == 2
    assert thresholds[0] >= 0.05


def test_find_threshold_by_gaussian_fit_parallel():
    F_detrended = np.random.randn(500) * 0.1
    ll = [F_detrended, 5]
    thresh = find_threshold_by_gaussian_fit_parallel(ll, 0.99, 0.05, maximum_sigma=100)
    assert thresh >= 0.05


def test_find_threshold_by_gaussian_fit_parallel_tight():
    F_detrended = np.random.randn(500) * 3
    ll = [F_detrended, 0]
    thresh = find_threshold_by_gaussian_fit_parallel(ll, 0.99, 0.05, maximum_sigma=0.01)
    assert thresh == 1
