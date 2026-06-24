import os
import pytest
import numpy as np
from binarize2pcalcium.io import load_suite2p, load_inscopix

def test_load_suite2p(tmp_path):
    d = tmp_path / "plane0"
    d.mkdir()
    np.save(d / "F.npy", np.zeros((2, 10)))
    np.save(d / "stat.npy", np.array([{'ypix': [1]}], dtype=object))
    np.save(d / "iscell.npy", np.array([[1, 0]], dtype=object))
    np.save(d / "spks.npy", np.zeros((2, 10)))
    np.save(d / "ops.npy", np.array({'Lx': 100}, dtype=object))

    F, stat, iscell, spks, ops = load_suite2p(str(d))
    assert F.shape == (2, 10)
    assert len(stat) == 1
    assert spks.shape == (2, 10)
    assert 'Lx' in ops

def test_load_suite2p_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_suite2p(str(tmp_path / "missing"))

def test_load_inscopix(tmp_path):
    import csv
    d = tmp_path / "inscopix"
    d.mkdir()
    with open(d / "data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([1.0, 2.0, 3.0])
        writer.writerow([4.0, 5.0, 6.0])

    data = load_inscopix(str(d))
    assert data.shape == (2, 3)
    assert np.allclose(data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

def test_load_inscopix_missing(tmp_path):
    d = tmp_path / "inscopix"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        load_inscopix(str(d))
