"""Single-function pipeline for calcium imaging binarization.

This module replaces the Calcium class with a simple function-based API.
Just pass your data matrix (or a path) and configuration, get binarized
results back.

Example:
    from binarize2pcalcium import binarize

    # Option 1: Pass a matrix directly
    result = binarize(F, sample_rate=30, data_type='2p')

    # Option 2: Load from file and binarize
    result = binarize(F, sample_rate=20, data_type='1p')

    # Access results
    print(result.onphase)   # binarized onphase events
    print(result.upphase)   # binarized upphase events
    print(result.dff)       # dF/F traces
"""

from dataclasses import dataclass, field
import numpy as np
import os
from tqdm import tqdm

from .preprocess import compute_dff, low_pass_filter, detrend_traces
from .thresholds import find_threshold_by_gaussian_fit
from .binarization import binarize_onphase, binarize_upphase
from .io import load_suite2p, load_inscopix


@dataclass
class BinarizationResult:
    """Container for all pipeline outputs.

    Attributes:
        F_raw: Raw fluorescence input [n_cells, n_timepoints].
        dff: dF/F traces [n_cells, n_timepoints].
        F_filtered: Lowpass filtered dF/F [n_cells, n_timepoints].
        F_detrended: Detrended fluorescence [n_cells, n_timepoints].
        onphase: Binarized onphase events [n_cells, n_timepoints].
        upphase: Binarized upphase events [n_cells, n_timepoints].
        thresholds: Per-cell threshold values.
        sample_rate: Sampling rate used (Hz).
    """

    F_raw: np.ndarray
    dff: np.ndarray
    F_filtered: np.ndarray
    F_detrended: np.ndarray
    onphase: np.ndarray
    upphase: np.ndarray
    thresholds: list[float] = field(default_factory=list)
    sample_rate: float = 30.0
    data_type: str = '2p'

    def save(self, filepath: str):
        """Save all results to a .npz file."""
        np.savez(
            filepath,
            F_raw=self.F_raw,
            dff=self.dff,
            F_filtered=self.F_filtered,
            F_detrended=self.F_detrended,
            onphase=self.onphase,
            upphase=self.upphase,
            thresholds=np.array(self.thresholds),
            sample_rate=self.sample_rate,
            data_type=self.data_type,
        )

    @classmethod
    def load(cls, filepath: str) -> "BinarizationResult":
        """Load results from a .npz file."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            F_raw=data['F_raw'],
            dff=data['dff'],
            F_filtered=data['F_filtered'],
            F_detrended=data['F_detrended'],
            onphase=data['onphase'],
            upphase=data['upphase'],
            thresholds=data['thresholds'].tolist(),
            sample_rate=float(data['sample_rate']),
            data_type=str(data['data_type']),
        )


def binarize(
    F: np.ndarray | str,
    sample_rate: float = 30.0,
    data_type: str = '2p',
    high_cutoff: float = 2.0,
    detrend_order: int = 1,
    percentile_threshold: float = 0.9999,
    dff_min: float = 0.05,
    min_width_onphase: int = 15,
    min_width_upphase: int = 7,
    use_upphase: bool = True,
    remove_ends: bool = True,
    verbose: bool = False,
    **kwargs,
) -> BinarizationResult:
    """Run the full calcium binarization pipeline in one call.

    This is the single entry point that replaces the Calcium class.
    Pass in raw fluorescence data (or a path to Suite2p/Inscopix data),
    and get back binarized event traces.

    Args:
        F: Raw fluorescence 2D array [n_cells, n_timepoints], or a path
           string to Suite2p 'plane0' directory or Inscopix CSV directory.
        sample_rate: Imaging sampling rate in Hz (default: 30 for 2P, 20 for 1P).
        data_type: '2p' for two-photon, '1p' for Inscopix one-photon.
        high_cutoff: Lowpass filter cutoff in Hz (default: 2.0).
        detrend_order: Polynomial order for detrending (default: 1 = linear).
        percentile_threshold: Gaussian-fit percentile for threshold (default: 0.9999).
        dff_min: Minimum dF/F threshold (default: 0.05).
        min_width_onphase: Minimum event width in samples for onphase (default: sample_rate/2).
        min_width_upphase: Minimum event width in samples for upphase (default: sample_rate/4).
        use_upphase: Whether to compute upphase binarization (default: True).
        remove_ends: Replace first/last 300 samples with noise to remove edge artifacts.
        verbose: Print progress information.

    Returns:
        BinarizationResult with all pipeline outputs.

    Example:
        # From a data matrix:
        F = np.load('cell_traces.npy')   # shape: [n_cells, n_timepoints]
        result = binarize(F, sample_rate=30, data_type='2p')
        print(result.onphase.shape)

        # From Suite2p output:
        result = binarize('/path/to/suite2p/plane0/', data_type='2p')
    """
    if verbose:
        print(f"Running binarization pipeline:")
        print(f"  data_type={data_type}, sample_rate={sample_rate} Hz")
        print(f"  high_cutoff={high_cutoff} Hz, detrend_order={detrend_order}")

    # Step 1: Load data if path is given
    if isinstance(F, str):
        data_path = F
        if data_type == '2p':
            F_raw, stat, iscell, spks, ops = load_suite2p(data_path)
        elif data_type == '1p':
            F_raw = load_inscopix(data_path)
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use '2p' or '1p'.")
    else:
        F_raw = np.asarray(F, dtype=np.float64)

    if verbose:
        print(f"  Input shape: {F_raw.shape}")

    n_cells, n_timepoints = F_raw.shape

    # Set default widths based on sample rate if not explicitly given
    if min_width_onphase is None:
        min_width_onphase = int(sample_rate // 2)
    if min_width_upphase is None:
        min_width_upphase = int(sample_rate // 4)

    # Step 2: Compute dF/F
    dff = compute_dff(F_raw, data_type=data_type)

    # Step 3: Lowpass filter
    # Default filter order = 1 matches the original Calcium class logic
    F_filtered = low_pass_filter(dff, high_cutoff, sample_rate, order=1)
    # Save a copy *before* detrending — the original uses F_filtered for upphase
    F_filtered_saved = F_filtered.copy()

    # Step 4: Remove edge artifacts (optional)
    if remove_ends and n_timepoints > 600:
        F_filtered[:, :300] = (np.random.rand(n_cells, 300) - 0.5) / 100
        F_filtered[:, -300:] = (np.random.rand(n_cells, 300) - 0.5) / 100

    # Step 5: Detrend
    F_detrended = detrend_traces(
        F_filtered, sample_rate, detrend_model_order=detrend_order
    )

    # Step 6: Compute thresholds from detrended data
    if verbose:
        print("  Computing thresholds...")
    thresholds = find_threshold_by_gaussian_fit(
        F_detrended, percentile_threshold, dff_min
    )

    # Step 7: Binarize onphase from detrended data
    if verbose:
        print("  Binarizing onphase...")
    onphase = binarize_onphase(F_detrended, thresholds, min_width_onphase)

    # Step 8: Binarize upphase — use F_filtered_saved (pre-detrend),
    #         but compute the gradient from F_detrended (original logic)
    if use_upphase:
        if verbose:
            print("  Binarizing upphase...")
        upphase = binarize_upphase(
            F_filtered_saved,
            thresholds,
            min_width_upphase,
            der_min_slope=0,
            F_detrended=F_detrended,
        )
    else:
        upphase = np.zeros_like(F_detrended)

    if verbose:
        n_on = np.sum(onphase > 0)
        n_up = np.sum(upphase > 0)
        print(f"  Done. Onphase events: {n_on}, Upphase events: {n_up}")

    return BinarizationResult(
        F_raw=F_raw,
        dff=dff,
        F_filtered=F_filtered_saved,
        F_detrended=F_detrended,
        onphase=onphase,
        upphase=upphase,
        thresholds=thresholds,
        sample_rate=sample_rate,
        data_type=data_type,
    )
