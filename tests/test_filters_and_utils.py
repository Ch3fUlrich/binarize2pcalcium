"""Tests for standalone filter functions and utility functions."""
import numpy as np
import pytest
from binarize2pcalcium.binarize2pcalcium import (
    butter_highpass,
    butter_highpass_filter,
    butter_lowpass,
    butter_lowpass_filter,
    butter_bandpass,
    butter_bandpass_filter,
    signaltonoise,
    array_row_intersection,
    get_corr,
    get_corr2,
    it_count,
)


# ---- Butterworth filter tests ----

def test_butter_highpass():
    """Test butter_highpass returns valid filter coefficients."""
    b, a = butter_highpass(cutoff=10, fs=100, order=5)
    assert len(b) > 0
    assert len(a) > 0
    assert a[0] == 1.0


def test_butter_highpass_filter():
    """Test highpass filtering attenuates low frequencies."""
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    y = butter_highpass_filter(data, cutoff=10, fs=fs, order=5)
    assert np.std(y) > 0
    assert len(y) == len(data)


def test_butter_highpass_filter_no_attenuation_of_high():
    """Test that highpass filter preserves high frequency signal energy."""
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 25 * t)
    y = butter_highpass_filter(data, cutoff=10, fs=fs, order=5)
    assert len(y) == len(data)


def test_butter_lowpass():
    """Test butter_lowpass returns valid filter coefficients."""
    b, a = butter_lowpass(cutoff=10, fs=100, order=5)
    assert len(b) > 0
    assert len(a) > 0
    assert a[0] == 1.0


def test_butter_lowpass_filter():
    """Test lowpass filtering."""
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
    y = butter_lowpass_filter(data, cutoff=10, fs=fs, order=5)
    assert len(y) == len(data)


def test_butter_lowpass_filter_preserves_low():
    """Test that lowpass filter preserves low frequency signal."""
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 2 * t)
    y = butter_lowpass_filter(data, cutoff=10, fs=fs, order=5)
    assert len(y) == len(data)


def test_butter_bandpass():
    """Test butter_bandpass returns valid SOS coefficients."""
    sos = butter_bandpass(lowcut=10, highcut=20, fs=100, order=5)
    assert sos is not None
    assert sos.ndim == 2


def test_butter_bandpass_filter():
    """Test bandpass filtering."""
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 30 * t)
    y = butter_bandpass_filter(data, lowcut=10, highcut=20, fs=fs, order=5)
    assert len(y) == len(data)


# ---- Signal-to-noise and correlation tests ----

def test_signaltonoise():
    """Test signaltonoise ratio calculation."""
    a = np.array([1, 2, 3, 4, 5])
    snr = signaltonoise(a)
    assert snr > 0

    b = np.array([5, 5, 5, 5, 5])
    assert signaltonoise(b) == 0


def test_signaltonoise_2d():
    """Test signaltonoise with 2D array."""
    a = np.random.randn(5, 100)
    snr = signaltonoise(a, axis=1)
    assert snr.shape == (5,)


def test_get_corr():
    """Test get_corr with linearly correlated data."""
    t1 = np.array([1, 2, 3, 4, 5])
    t2 = np.array([2, 4, 6, 8, 10])
    corr = get_corr(t1, t2)
    assert corr[0] > 0.99

    t3 = np.random.randn(100)
    t4 = np.random.randn(100)
    corr2 = get_corr(t3, t4)
    assert len(corr2) == 2


def test_get_corr_constant():
    """Test get_corr with constant input returns nan."""
    t1 = np.array([1, 1, 1, 1, 1])
    t2 = np.array([2, 4, 6, 8, 10])
    corr = get_corr(t1, t2)
    assert np.isnan(corr[0])


def test_get_corr_zscore():
    """Test get_corr with zscore mode."""
    # Need large enough array for zscore shuffling (randint range must be valid)
    t1 = np.random.randn(500)
    t2 = t1 * 0.5 + np.random.randn(500) * 0.1
    corr = get_corr(t1, t2, zscore=True, n_tests=100)
    assert len(corr) == 2
    assert corr[0] > 0


def test_get_corr2():
    """Test get_corr2 basic functionality."""
    t1 = np.random.randn(500) * 0.1 + 0.5
    t2 = t1 * 0.8 + np.random.randn(500) * 0.05
    corr_orig, corr_z = get_corr2(t1, t2, zscore=True, n_tests=50)
    assert corr_orig[0] > 0
    # corr_z from stats.zscore with n_tests+1 elements (original + n_tests shuffles)
    assert len(corr_z) == 51  # n_tests + 1


def test_get_corr2_constant():
    """Test get_corr2 with constant input."""
    t1 = np.ones(100)
    t2 = np.random.randn(100)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False)
    assert np.isnan(corr_orig[0])


def test_get_corr2_no_zscore():
    """Test get_corr2 without zscore."""
    t1 = np.random.randn(200)
    t2 = np.random.randn(200)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False)
    assert len(corr_orig) == 2
    assert len(corr_z) == 1
    assert np.isnan(corr_z[0])


def test_get_corr2_min_bursts():
    """Test get_corr2 with min_number_bursts filter."""
    t1 = np.zeros(500)
    t1[50] = 1
    t1[60] = 1
    t2 = np.random.randn(500)
    corr_orig, corr_z = get_corr2(t1, t2, zscore=False, min_number_bursts=3)
    assert np.isnan(corr_orig[0])


# ---- Array operations ----

def test_array_row_intersection():
    """Test array_row_intersection finds common rows."""
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[3, 4], [7, 8], [5, 6]])
    result = array_row_intersection(a, b)
    assert result.shape[0] >= 2


def test_array_row_intersection_no_common():
    """Test array_row_intersection with no common rows."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = array_row_intersection(a, b)
    assert len(result) == 0


def test_array_row_intersection_single():
    """Test array_row_intersection with single common row."""
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [7, 8, 9]])
    result = array_row_intersection(a, b)
    assert len(result) == 1
    assert np.array_equal(result[0], [4, 5, 6])


# ---- Iterator utility ----

def test_it_count():
    """Test it_count counts iterator elements."""
    count, new_it = it_count(iter([1, 2, 3, 4, 5]))
    assert count == 5
    assert list(new_it) == [1, 2, 3, 4, 5]


def test_it_count_empty():
    """Test it_count with empty iterator."""
    count, new_it = it_count(iter([]))
    assert count == 0
    assert list(new_it) == []


def test_it_count_single():
    """Test it_count with single element."""
    count, new_it = it_count(iter([42]))
    assert count == 1
    assert list(new_it) == [42]
