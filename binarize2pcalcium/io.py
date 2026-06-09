"""Data I/O functions for loading calcium imaging data.

Supports Suite2p and Inscopix data formats.
"""

import os
import numpy as np


def load_suite2p(data_dir: str) -> tuple:
    """Load Suite2p output from a standard directory structure.

    Expects: data_dir/stat.npy, data_dir/F.npy, data_dir/iscell.npy,
    data_dir/spks.npy, data_dir/ops.npy

    Args:
        data_dir: Path to Suite2p plane directory.

    Returns:
        Tuple of (F_raw, F_neuropil, stat, iscell, spks, ops).
    """
    import yaml

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Suite2p data directory not found: {data_dir}")

    F = np.load(os.path.join(data_dir, 'F.npy'))
    stat = np.load(os.path.join(data_dir, 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(data_dir, 'iscell.npy'), allow_pickle=True)
    spks = np.load(os.path.join(data_dir, 'spks.npy'))
    ops = np.load(os.path.join(data_dir, 'ops.npy'), allow_pickle=True).item()

    return F, stat, iscell, spks, ops


def load_inscopix(data_dir: str) -> np.ndarray:
    """Load Inscopix CNMF-E CSV data.

    Args:
        data_dir: Directory containing .csv file.

    Returns:
        2D fluorescence array [n_cells, n_timepoints].
    """
    import glob
    import csv

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {data_dir}")

    fname = csv_files[0]
    data = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])

    return np.array(data)
