"""binarize2pcalcium - Binarization pipeline for 2P and 1P calcium imaging.

Usage:
    from binarize2pcalcium import binarize, BinarizationResult

    # From a data matrix:
    result = binarize(F, sample_rate=30, data_type='2p')

    # From Suite2p data:
    result = binarize('/path/to/suite2p/plane0/', data_type='2p')

    # From Inscopix CSV:
    result = binarize('/path/to/inscopix/data/', data_type='1p', sample_rate=20)
"""

from .pipeline import binarize, BinarizationResult
from .filters import (
    butter_highpass,
    butter_highpass_filter,
    butter_lowpass,
    butter_lowpass_filter,
    butter_bandpass,
    butter_bandpass_filter,
    medfilt,
)
from .thresholds import (
    signaltonoise,
    find_threshold_by_gaussian_fit,
    find_threshold_by_gaussian_fit_parallel,
)
from .preprocess import (
    standardize,
    compute_dff,
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    detrend_traces,
    filter_model,
    wavelet_filter,
)
from .binarization import (
    binarize_onphase,
    binarize_std,
    binarize_derivative,
    scale_binarized,
    binarize_upphase,
    smooth_traces,
)
from .correlation import (
    get_corr,
    get_corr2,
    correlations_parallel,
    make_correlation_array,
    get_correlations,
)
from .overlap import (
    array_row_intersection,
    find_overlaps,
    find_overlaps1,
    find_overlaps2,
    make_overlap_database,
    find_inter_cell_distance,
    alpha_shape,
)
from .dedup import (
    it_count,
    del_highest_connected_nodes,
    del_highest_connected_nodes_without_corr,
    del_lowest_snr,
)
from .pca import (
    run_UMAP,
    compute_PCA,
    compute_TSNE,
    compute_UMAP,
    fit_curves_aucs,
    fit_curves_general,
    load_pca_animal,
)
from .io import load_suite2p, load_inscopix
from .data_simulation import (
    SimulationConfig,
    simulate_calcium_data,
    simulate_and_binarize,
)

# simulate_and_binarize now returns (result, event_gt, dff_true) — 3-tuple


__all__ = [
    # Pipeline (primary API)
    "binarize",
    "BinarizationResult",
    # Filters
    "butter_highpass",
    "butter_highpass_filter",
    "butter_lowpass",
    "butter_lowpass_filter",
    "butter_bandpass",
    "butter_bandpass_filter",
    "medfilt",
    # Thresholds
    "signaltonoise",
    "find_threshold_by_gaussian_fit",
    "find_threshold_by_gaussian_fit_parallel",
    # Preprocess
    "standardize",
    "compute_dff",
    "low_pass_filter",
    "high_pass_filter",
    "band_pass_filter",
    "detrend_traces",
    "filter_model",
    "wavelet_filter",
    # Binarization
    "binarize_onphase",
    "binarize_std",
    "binarize_derivative",
    "scale_binarized",
    "binarize_upphase",
    "smooth_traces",
    # Correlation
    "get_corr",
    "get_corr2",
    "correlations_parallel",
    "make_correlation_array",
    "get_correlations",
    # Overlap
    "array_row_intersection",
    "find_overlaps",
    "find_overlaps1",
    "find_overlaps2",
    "make_overlap_database",
    "find_inter_cell_distance",
    "alpha_shape",
    # Dedup
    "it_count",
    "del_highest_connected_nodes",
    "del_highest_connected_nodes_without_corr",
    "del_lowest_snr",
    # PCA
    "run_UMAP",
    "compute_PCA",
    "compute_TSNE",
    "compute_UMAP",
    "fit_curves_aucs",
    "fit_curves_general",
    "load_pca_animal",
    # I/O
    "load_suite2p",
    "load_inscopix",
]
