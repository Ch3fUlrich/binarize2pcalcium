"""Tests for filters module."""
import numpy as np
import pytest
from binarize2pcalcium.filters import (
    butter_highpass,
    butter_highpass_filter,
    butter_lowpass,
    butter_lowpass_filter,
    butter_bandpass,
    butter_bandpass_filter,
    medfilt,
)


def test_butter_highpass():
    b, a = butter_highpass(cutoff=10, fs=100, order=5)
    assert len(b) > 0
    assert len(a) > 0
    assert a[0] == 1.0


def test_butter_highpass_filter():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    y = butter_highpass_filter(data, cutoff=10, fs=fs, order=5)
    assert np.std(y) > 0
    assert len(y) == len(data)


def test_butter_lowpass():
    b, a = butter_lowpass(cutoff=10, fs=100, order=5)
    assert len(b) > 0
    assert len(a) > 0
    assert a[0] == 1.0


def test_butter_lowpass_filter():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    y = butter_lowpass_filter(data, cutoff=10, fs=fs, order=5)
    assert len(y) == len(data)


def test_butter_bandpass():
    sos = butter_bandpass(lowcut=10, highcut=20, fs=100, order=5)
    assert sos is not None
    assert sos.ndim == 2


def test_butter_bandpass_filter():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 30 * t)
    y = butter_bandpass_filter(data, lowcut=10, highcut=20, fs=fs, order=5)
    assert len(y) == len(data)


def test_medfilt():
    trace = np.random.randn(100)
    filtered = medfilt(trace, 5)
    assert len(filtered) == len(trace)


def test_medfilt_kernel_validation():
    with pytest.raises(AssertionError, match="Median filter length must be odd"):
        medfilt(np.random.randn(100), 4)


def test_medfilt_dim_validation():
    with pytest.raises(AssertionError, match="Input must be one-dimensional"):
        medfilt(np.random.randn(10, 100), 5)
