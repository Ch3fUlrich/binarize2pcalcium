"""Core binarization functions for calcium fluorescence traces.

Converts continuous fluorescence traces into binary event representations
using threshold-based detection with minimum width criteria.
"""

import numpy as np
from tqdm import trange
from scipy.signal import find_peaks, peak_widths
from .filters import butter_lowpass_filter


def binarize_onphase(
    traces: np.ndarray,
    thresholds: list[float],
    min_width_event: int = 15,
    text: str = '',
) -> np.ndarray:
    """Binarize traces by thresholding: values >= threshold become 1.

    Short events (below min_width_event samples) are removed.

    Args:
        traces: 2D array [n_cells, n_timepoints] of filtered fluorescence.
        thresholds: Per-cell threshold values.
        min_width_event: Minimum width (in samples) of valid events.

    Returns:
        Binarized array of same shape (0 = no event, 1 = event).
    """
    traces_bin = traces.copy()
    for k in trange(
        traces.shape[0], desc='binarizing continuous traces'+text, position=0, leave=True
    ):
        temp = traces[k].copy()
        thresh_local = thresholds[k]

        idx1 = np.where(temp >= thresh_local)[0]
        temp = temp * 0
        temp[idx1] = 1

        peaks, _ = find_peaks(temp)
        widths, heights, starts, ends = peak_widths(temp, peaks)

        xys = np.int32(np.vstack((starts, ends)).T)
        idx = np.where(widths < min_width_event)[0]
        xys = np.delete(xys, idx, axis=0)

        traces_bin[k] = traces_bin[k] * 0
        for p in range(xys.shape[0]):
            traces_bin[k, xys[p, 0] : xys[p, 1]] = 1

    return traces_bin


def binarize_std(
    traces: np.ndarray, thresh: float = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Binarize by standard-deviation threshold: value >= std*thresh is an event.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        thresh: Multiplier for std to set threshold.

    Returns:
        Tuple of (binarized_traces, anti_aliased_traces).
    """
    traces_out = traces.copy() * 0
    traces_out_anti_aliased = traces.copy() * 0
    for k in trange(traces.shape[0], desc='binarizing'):
        temp = traces[k]
        std = np.std(temp)
        idx = np.where(temp >= std * thresh)[0]

        traces_out[k] = 0
        traces_out[k, idx] = 1

        for id_ in idx:
            traces_out_anti_aliased[k, id_ : id_ + 20] = 1
            if k > 0:
                traces_out_anti_aliased[k - 1, id_ : id_ + 20] = 1

    return traces_out, traces_out_anti_aliased


def binarize_derivative(
    traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivative of traces for event detection.

    Args:
        traces: 2D array [n_cells, n_timepoints].

    Returns:
        Tuple of (gradient_traces, anti_aliased_traces).
    """
    traces_out = traces.copy() * 0
    traces_out_anti_aliased = traces.copy() * 0
    for k in trange(traces.shape[0], desc='computing derivative'):
        grad = np.gradient(traces[k])
        traces_out[k] = grad

    return traces_out, traces_out_anti_aliased


def scale_binarized(
    traces: np.ndarray, traces_scale: np.ndarray, min_event_amplitude: float = 0
) -> np.ndarray:
    """Scale binarized events by the sum of their corresponding DFF values.

    Each event's amplitude is replaced by the sum of scaled values within
    the event window plus a buffer.

    Args:
        traces: Binarized 2D array [n_cells, n_timepoints].
        traces_scale: DFF-scaled 2D array of same shape.
        min_event_amplitude: Minimum amplitude; events below this are zeroed.

    Returns:
        Scaled binarized array.
    """
    for k in trange(traces.shape[0], desc='scaling binarized data'):
        temp = traces[k].copy()
        peaks, _ = find_peaks(temp)
        widths, heights, starts, ends = peak_widths(temp, peaks)

        xys = np.int32(np.vstack((starts, ends)).T)
        buffer = 5
        for t in range(xys.shape[0]):
            peak = np.sum(traces_scale[k, xys[t, 0] : xys[t, 1] + buffer])
            temp[xys[t, 0] : xys[t, 1]] *= peak
            if np.max(temp[xys[t, 0] : xys[t, 1]]) < min_event_amplitude:
                temp[xys[t, 0] : xys[t, 1] + 1] = 0

        traces[k] = temp

    return traces


def binarize_upphase(
    F_filtered: np.ndarray,
    thresholds: list[float],
    min_width_event: int = 7,
    der_min_slope: float = 0,
    F_detrended: np.ndarray | None = None,
) -> np.ndarray:
    """Binarize the rising phase (upphase) of fluorescence traces.

    Computes the gradient from F_detrended (or F_filtered if not provided),
    zeroes portions with derivative <= der_min_slope, then applies onphase
    binarization.  Matches the original Calcium class logic.

    Args:
        F_filtered: 2D array [n_cells, n_timepoints] of lowpass-filtered dF/F
                    (pre-detrend, used for binarization itself).
        thresholds: Per-cell threshold values.
        min_width_event: Minimum event width in samples.
        der_min_slope: Minimum slope; values with derivative <= this are zeroed.
        F_detrended: 2D array [n_cells, n_timepoints] of detrended traces
                     used to compute the gradient.  Defaults to F_filtered.

    Returns:
        Binarized upphase array.
    """
    if F_detrended is None:
        F_detrended = F_filtered
    der = np.float32(np.gradient(F_detrended, axis=1))
    idx = np.where(der <= der_min_slope)
    F_upphase = F_filtered.copy()
    F_upphase[idx] = 0

    return binarize_onphase(F_upphase, thresholds, min_width_event, " filtered fluorescence upphase")


def smooth_traces(
    traces: np.ndarray, sample_rate: float
) -> np.ndarray:
    """Smooth traces with exponential convolution and lowpass filter.

    Args:
        traces: 2D array [n_cells, n_timepoints].
        sample_rate: Sampling frequency in Hz.

    Returns:
        Smoothed array of same shape.
    """
    import scipy.signal as scisig

    M = 100
    tau = 100
    d_exp = scisig.windows.exponential(M, 0, tau, False)

    traces_out = traces.copy()
    for k in trange(traces.shape[0], desc='convolving with exponential and filtering'):
        temp = traces_out[k].copy()
        temp = np.convolve(temp, d_exp, mode='full')[: temp.shape[0]]
        temp = butter_lowpass_filter(temp, 2, 30)
        traces_out[k] = temp

    return traces_out
