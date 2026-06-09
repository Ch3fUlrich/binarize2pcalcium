"""Correlation computation for calcium imaging data.

Functions for computing Pearson correlations between fluorescence traces
and binarized rasters, including parallel and z-score support.
"""

import os
import numpy as np
import scipy.stats
from tqdm import trange


def get_corr(
    temp1: np.ndarray,
    temp2: np.ndarray,
    zscore: bool = False,
    n_tests: int = 500,
) -> list:
    """Compute Pearson correlation between two 1D arrays.

    Args:
        temp1: First 1D array.
        temp2: Second 1D array.
        zscore: If True, compute z-score via shuffling.
        n_tests: Number of shuffles for z-score.

    Returns:
        List of [correlation, p_value] or [correlation, z_score] if zscore=True.
    """
    if np.all(temp1 == temp1[0]):
        return [np.nan, 1]
    if np.all(temp2 == temp2[0]):
        return [np.nan, 1]

    corr = scipy.stats.pearsonr(temp1, temp2)

    if zscore:
        corr_s = []
        for _ in range(n_tests):
            idx = np.random.randint(100, temp2.shape[0] - 100)
            temp2_shuffled = np.roll(temp2, idx)
            corr_s.append(scipy.stats.pearsonr(temp1, temp2_shuffled)[0])
        corr_s = np.array(corr_s)
        corr_z = (corr[0] - np.mean(corr_s)) / np.std(corr_s)
        corr = [corr[0], corr_z]

    return corr


def get_corr2(
    temp1: np.ndarray,
    temp2: np.ndarray,
    zscore: bool,
    n_tests: int = 1000,
    min_number_bursts: int = 0,
) -> tuple:
    """Compute Pearson correlation with optional z-score and burst filtering.

    Args:
        temp1: First 1D array.
        temp2: Second 1D array.
        zscore: If True, compute z-score via shuffling.
        n_tests: Number of shuffles for z-score.
        min_number_bursts: Minimum number of non-zero entries required.

    Returns:
        Tuple of (corr_original, corr_z_array).
        corr_original is [correlation, p_value].
        corr_z_array is z-score array or [np.nan].
    """
    if len(np.unique(temp1)) == 1 or len(np.unique(temp2)) == 1:
        return [np.nan, 1], [np.nan]

    if np.sum(temp1 != 0) < min_number_bursts or np.sum(temp2 != 0) < min_number_bursts:
        return [np.nan, 1], [np.nan]

    corr_original = scipy.stats.pearsonr(temp1, temp2)

    corr_array = []
    corr_array.append(corr_original[0])

    if zscore:
        corr_s = []
        for _ in range(n_tests):
            idx = np.random.randint(-temp2.shape[0], temp2.shape[0], 1)
            temp2_shuffled = np.roll(temp2, idx)
            corr_s_item = scipy.stats.pearsonr(temp1, temp2_shuffled)
            corr_array.append(corr_s_item[0])
        corr_z = scipy.stats.zscore(corr_array)
    else:
        corr_z = [np.nan]

    return corr_original, corr_z


def correlations_parallel(
    ids: list[int],
    rasters: np.ndarray,
    rasters_DFF: np.ndarray,
    binning_window: int = 30,
    subsample: int = 5,
    scale_by_DFF: bool = True,
    zscore: bool = False,
) -> list:
    """Compute pairwise correlations for specified cell IDs.

    Args:
        ids: List of cell IDs to compute correlations for.
        rasters: Binarized raster array [n_cells, n_timepoints].
        rasters_DFF: DFF-scaled raster array [n_cells, n_timepoints].
        binning_window: Number of samples to bin together.
        subsample: Subsampling rate (every Nth sample).
        scale_by_DFF: Whether to multiply by DFF values.
        zscore: Whether to compute z-scores.

    Returns:
        List of [cell_id_1, cell_id_2, corr, p_value] entries.
    """
    corrs = []
    for k in ids:
        temp1 = rasters[k][::subsample]
        if scale_by_DFF:
            temp1 = temp1 * rasters_DFF[k][::subsample]

        if binning_window != 1:
            tt = []
            for q in range(0, temp1.shape[0], binning_window):
                tt.append(np.sum(temp1[q : q + binning_window]))
            temp1 = np.array(tt)

        for p in range(rasters.shape[0]):
            temp2 = rasters[p][::subsample]
            if scale_by_DFF:
                temp2 = temp2 * rasters_DFF[p][::subsample]

            if binning_window != 1:
                tt = []
                for q in range(0, temp2.shape[0], binning_window):
                    tt.append(np.sum(temp2[q : q + binning_window]))
                temp2 = np.array(tt)

            corr, corr_z = get_corr2(temp1, temp2, zscore)
            corrs.append([k, p, corr[0], corr[1]])

    return corrs


def make_correlation_array(corrs: list, n_cells: int) -> np.ndarray:
    """Build 3D correlation array from list of correlation entries.

    Args:
        corrs: List of [cell1, cell2, pearson_corr, p_value] entries.
        n_cells: Total number of cells.

    Returns:
        3D array of shape [n_cells, n_cells, 2] where [...,0] = corr, [...,1] = pval.
    """
    corr_array = np.zeros((n_cells, n_cells, 2), 'float32')

    for k in trange(len(corrs)):
        cell1 = int(corrs[k][0])
        cell2 = int(corrs[k][1])
        pcor = corrs[k][2]
        pval = corrs[k][3]

        corr_array[cell1, cell2, 0] = pcor
        corr_array[cell1, cell2, 1] = pval

    return corr_array


def get_correlations(ids: np.ndarray, corr_array: np.ndarray) -> np.ndarray:
    """Extract correlation values for given cell IDs from a correlation array.

    Args:
        ids: 1D array of cell IDs.
        corr_array: 3D correlation array [n_cells, n_cells, 2].

    Returns:
        1D array of correlation values for all unique pairs.
    """
    corrs = []
    for i in range(ids.shape[0]):
        for ii in range(i + 1, ids.shape[0], 1):
            if ids[i] < ids[ii]:
                corrs.append(corr_array[ids[i], ids[ii], 0])
            else:
                corrs.append(corr_array[ids[ii], ids[i], 0])
    corrs = np.array(corrs)
    return corrs
