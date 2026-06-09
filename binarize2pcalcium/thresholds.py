"""Threshold computation for calcium imaging binarization.

Functions for determining event-detection thresholds from fluorescence
trace statistics using Gaussian fitting.
"""

import numpy as np
import scipy.stats
from statistics import NormalDist
from tqdm import trange


def signaltonoise(a: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
    """Compute signal-to-noise ratio: mean / std.

    Args:
        a: Input array.
        axis: Axis along which to compute (default 0).
        ddof: Delta degrees of freedom for std (default 0).

    Returns:
        SNR values. Returns 0 where std is 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def find_threshold_by_gaussian_fit(
    F_filtered: np.ndarray,
    percentile_threshold: float,
    dff_min: float,
) -> list[float]:
    """Find per-cell thresholds by fitting a Gaussian to the lower half
    of the fluorescence distribution centered on the mode.

    Uses mode mirroring: values below the mode are mirrored, a Gaussian
    is fit to the pooled distribution, and the threshold is set at the
    given cumulative probability.

    Args:
        F_filtered: 2D array of filtered fluorescence [n_cells, n_timepoints].
        percentile_threshold: Cumulative probability threshold (e.g., 0.9999).
        dff_min: Minimum dF/F threshold to enforce per cell.

    Returns:
        List of per-cell thresholds.
    """
    thresholds = []
    for k in trange(F_filtered.shape[0], desc='fitting mode to physics'):
        F_filtered2 = F_filtered[k]

        y_mode = scipy.stats.mode(F_filtered2, keepdims=False)
        if hasattr(y_mode, 'mode'):
            y_mode = y_mode.mode
        else:
            y_mode = y_mode[0]

        idx = np.where(F_filtered2 <= y_mode)[0]
        pts_neg = F_filtered2[idx]
        pts_pos = -pts_neg.copy()

        pooled = np.hstack((pts_neg, pts_pos))

        norm = NormalDist.from_samples(pooled)
        mu = norm.mean
        sigma = norm.stdev

        x = np.arange(-8, 16, 0.0001)
        y_fit = scipy.stats.norm.pdf(x, mu, sigma)
        y_fit = y_fit / np.max(y_fit)

        cumsum = np.cumsum(y_fit)
        cumsum = cumsum / np.max(cumsum)

        idx = np.where(cumsum > percentile_threshold)[0]
        thresh = x[idx[0]]

        thresh_max = max(thresh, dff_min)
        thresholds.append(thresh_max)

    return thresholds


def find_threshold_by_gaussian_fit_parallel(
    ll: list,
    percentile_threshold: float,
    snr_min: float,
    maximum_sigma: float = 100,
) -> float:
    """Find threshold for a single cell via Gaussian fitting (parallel worker).

    Fits a Gaussian to mode-centered fluorescence distribution and
    returns the threshold at the given percentile.

    Args:
        ll: [F_detrended_array, cell_id] pair.
        percentile_threshold: Cumulative probability threshold.
        snr_min: Minimum threshold value to enforce.
        maximum_sigma: If fitted sigma exceeds this, threshold is set to 1.

    Returns:
        Computed threshold value.
    """
    cell_id = ll[1]
    F_detrended = ll[0]

    try:
        y = np.histogram(F_detrended, bins=np.arange(-25, 25, 0.001))
        y_mode = y[1][np.argmax(y[0])]

        idx = np.where(F_detrended <= y_mode)[0]
        pts_neg = F_detrended[idx]
        pts_pos = -pts_neg.copy()

        pooled = np.hstack((pts_neg, pts_pos))

        norm = NormalDist.from_samples(pooled)
        mu = norm.mean
        sigma = norm.stdev

        x = np.arange(-25, 25, 0.001)
        y_fit = scipy.stats.norm.pdf(x, mu, sigma)
        y_fit = y_fit / np.max(y_fit)

        cumsum = np.cumsum(y_fit)
        cumsum = cumsum / np.max(cumsum)

        idx = np.where(cumsum > percentile_threshold)[0]
        thresh = x[idx[0]]

        if sigma > maximum_sigma:
            thresh = 1

    except Exception:
        print("error data corrupt: data: ", F_detrended)
        thresh = 0

    thresh_max = max(thresh, snr_min)
    return thresh_max
