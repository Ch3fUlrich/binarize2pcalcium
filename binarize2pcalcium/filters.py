"""Signal processing filters for calcium imaging data.

Provides Butterworth filter design and application functions for
highpass, lowpass, and bandpass filtering of fluorescence traces.
"""

import numpy as np
from scipy.signal import butter, sosfilt, filtfilt


def butter_highpass(cutoff: float, fs: float, order: int = 5):
    """Design a Butterworth highpass filter.

    Args:
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 5
) -> np.ndarray:
    """Apply a Butterworth highpass filter to 1D data.

    Args:
        data: 1D array of fluorescence values.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Filtered array of same shape as input.
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    """Design a Butterworth lowpass filter.

    Args:
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Tuple of (b, a) filter coefficients.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 5
) -> np.ndarray:
    """Apply a Butterworth lowpass filter to 1D data.

    Args:
        data: 1D array of fluorescence values.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Filtered array of same shape as input.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Design a Butterworth bandpass filter (SOS form).

    Args:
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        SOS (second-order sections) filter coefficients.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to 1D data.

    Args:
        data: 1D array of fluorescence values.
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 5).

    Returns:
        Filtered array of same shape as input.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def medfilt(x: np.ndarray, k: int) -> np.ndarray:
    """Apply a length-k median filter to a 1D array.

    Boundaries are extended by repeating endpoints.

    Args:
        x: 1D input array.
        k: Filter kernel size (must be odd).

    Returns:
        Median-filtered 1D array.

    Raises:
        AssertionError: If k is not odd or x is not 1D.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)
