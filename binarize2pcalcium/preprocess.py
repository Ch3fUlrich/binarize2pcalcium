"""Preprocessing of calcium fluorescence traces.

Functions for standardization, detrending, and trace-level filtering
prior to binarization.
"""

import numpy as np
import scipy.signal
from tqdm import trange
from .filters import (
    butter_lowpass_filter,
    butter_highpass_filter,
    butter_bandpass_filter,
)


def standardize(traces: np.ndarray) -> np.ndarray:
    """Standardize traces per cell to [0, 1] range.

    Subtracts median and divides by range.

    Args:
        traces: 2D array [n_cells, n_timepoints].

    Returns:
        Standardized array of same shape.
    """
    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='standardizing'):
        temp = traces[k]
        temp -= np.median(temp)
        temp = temp / (np.max(temp) - np.min(temp))
        traces_out[k] = temp
    return traces_out


def compute_dff(F: np.ndarray, data_type: str = '2p') -> np.ndarray:
    """Compute dF/F from raw fluorescence.

    Args:
        F: Raw fluorescence 2D array [n_cells, n_timepoints].
        data_type: '2p' or '1p'. For 1p, uses F - F0.

    Returns:
        dF/F array of same shape.
    """
    f0s = np.abs(np.median(F, axis=1))
    if data_type == '1p':
        dff = F - f0s[:, None]
    else:
        dff = (F - f0s[:, None]) / f0s[:, None]
    return dff


def low_pass_filter(
    traces: np.ndarray, high_cutoff: float, sample_rate: float, order: int = 1
) -> np.ndarray:
    """Apply lowpass filter to each cell's trace.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        high_cutoff: Cutoff frequency in Hz.
        sample_rate: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered array of same shape.
    """
    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='low pass filter'):
        traces_out[k] = butter_lowpass_filter(
            traces[k], high_cutoff, sample_rate, order=order
        )
    return traces_out


def high_pass_filter(
    traces: np.ndarray, low_cutoff: float, sample_rate: float, order: int = 1
) -> np.ndarray:
    """Apply highpass filter to each cell's trace.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        low_cutoff: Cutoff frequency in Hz.
        sample_rate: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered array of same shape.
    """
    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='high pass filter'):
        traces_out[k] = butter_highpass_filter(
            traces[k], low_cutoff, sample_rate, order=order
        )
    return traces_out


def band_pass_filter(
    traces: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: float,
    order: int = 1,
) -> np.ndarray:
    """Apply bandpass (Chebyshev) filter to each cell's trace.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        low_cutoff: Low cutoff in Hz.
        high_cutoff: High cutoff in Hz.
        sample_rate: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered array of same shape.
    """
    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='band pass filter'):
        traces_out[k] = butter_bandpass_filter(
            traces[k], low_cutoff, high_cutoff, sample_rate, order=order
        )
    return traces_out


def detrend_traces(
    traces: np.ndarray,
    sample_rate: float,
    detrend_model_order: int = 1,
    detrend_filter_threshold: float = 0.001,
    mode_window: int | None = None,
) -> np.ndarray:
    """Detrend fluorescence traces by removing polynomial fit.

    Fits a polynomial to a very-lowpass filtered version of each trace
    and subtracts it. Also optionally does mode-based baseline removal.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        sample_rate: Sampling frequency in Hz.
        detrend_model_order: Polynomial order for trend fit (1=linear).
        detrend_filter_threshold: Lowpass cutoff for trend extraction.
        mode_window: Window size for piecewise mode subtraction (None = global).

    Returns:
        Detrended array of same shape.
    """
    traces_out = traces.copy()
    t = np.arange(traces[0].shape[0]) if traces.shape[0] > 0 else None

    for k in trange(
        traces.shape[0],
        desc='model filter: remove bleaching or trends',
        position=0,
        leave=True,
    ):
        temp = traces[k]
        F_very_low_band_pass = butter_lowpass_filter(
            temp, detrend_filter_threshold, sample_rate, detrend_model_order
        )
        t01 = np.arange(F_very_low_band_pass.shape[0])

        if detrend_model_order == 1:
            z = np.polyfit(t01, F_very_low_band_pass, 1)
            p = np.poly1d(z)
            temp = temp - p(t)
            traces_out[k] = traces_out[k] - p(t)

        elif detrend_model_order > 1:
            z = np.polyfit(t01, F_very_low_band_pass, detrend_model_order)
            p = np.poly1d(z)
            temp = temp - p(t)
            traces_out[k] = traces_out[k] - p(t)

        # Mode-based baseline removal
        if mode_window is None:
            y = np.histogram(temp, bins=np.arange(-1, 1, 0.001))
            y_mode = y[1][np.argmax(y[0])]
            temp = temp - y_mode
        else:
            for q in range(0, temp.shape[0], mode_window):
                y = np.histogram(
                    temp[q : q + mode_window], bins=np.arange(-5, 5, 0.001)
                )
                y_mode = y[1][np.argmax(y[0])]
                temp[q : q + mode_window] = temp[q : q + mode_window] - y_mode

        traces_out[k] = temp

    return traces_out


def filter_model(
    traces: np.ndarray,
    sample_rate: float,
    detrend_model_order: int = 1,
) -> np.ndarray:
    """Filter traces using trend subtraction based on endpoint medians.

    For order 1: fits line through median of first/last 10K points.
    For order 2: fits quadratic to trace.
    For order > 2: lowpass filter then polynomial fit.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        sample_rate: Sampling frequency in Hz.
        detrend_model_order: Polynomial order for detrending.

    Returns:
        Filtered array of same shape.
    """
    traces_out = traces.copy()
    t = np.arange(traces[0].shape[0])

    for k in trange(traces.shape[0], desc='model filter: remove bleaching or trends'):
        temp = traces[k]

        if detrend_model_order == 1:
            median01 = np.array(
                [np.median(temp[:10000]), np.median(temp[-10000:])]
            )
            t01 = np.array([0, temp.shape[0] - 1])
            z = np.polyfit(t01, median01, 1)
            p = np.poly1d(z)
            temp = temp - p(t)
            traces_out[k] = traces_out[k] - p(t)

        elif detrend_model_order == 2:
            z = np.polyfit(t, temp, 2)
            p = np.poly1d(z)
            traces_out[k] = traces_out[k] - p(t)

        elif detrend_model_order > 2:
            temp_filt = butter_lowpass_filter(temp, 0.01, sample_rate, order=5)
            z = np.polyfit(t, temp_filt, detrend_model_order)
            p = np.poly1d(z)
            traces_out[k] = traces_out[k] - p(t)

    return traces_out


def wavelet_filter(traces: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply wavelet-based denoising filter to traces.

    Uses db3 wavelet decomposition, zeroes approximation coefficients,
    then subtracts reconstructed detail from original.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        sample_rate: Sampling frequency (unused, kept for API consistency).

    Returns:
        Filtered array of same shape.
    """
    import pywt

    def _wavelet(data, wname="db2", maxlevel=6):
        w = pywt.Wavelet('db3')
        c = pywt.wavedec(data, wname, level=maxlevel)
        c[0] = None  # Remove approximation coefficients
        data = pywt.waverec(c, wname)
        return data

    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='wavelet filter'):
        temp = traces[k]
        temp2 = _wavelet(temp)
        temp = temp - temp2
        traces_out[k] = temp

    return traces_out
