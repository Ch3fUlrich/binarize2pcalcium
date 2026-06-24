import pytest
import numpy as np
import os
from binarize2pcalcium.pipeline import binarize, BinarizationResult
from binarize2pcalcium.data_simulation import SimulationConfig, simulate_calcium_data

def test_binarize_str_input(tmp_path):
    F = np.random.randn(5, 100) * 0.1 + 0.5
    d = tmp_path / "plane0"
    d.mkdir()
    np.save(d / "F.npy", F)
    np.save(d / "stat.npy", np.array([{'ypix': [1]}] * 5, dtype=object))
    np.save(d / "iscell.npy", np.array([[1, 0]] * 5, dtype=object))
    np.save(d / "spks.npy", np.zeros((5, 100)))
    np.save(d / "ops.npy", np.array({'Lx': 100}, dtype=object))

    result = binarize(str(d), verbose=False)
    assert result.F_raw.shape == (5, 100)
    assert result.dff.shape == (5, 100)

def test_binarize_str_input_yaml(tmp_path):
    d = tmp_path / "plane0"
    d.mkdir()
    F = np.random.randn(5, 100) * 0.1 + 0.5
    np.save(d / "F.npy", F)
    np.save(d / "stat.npy", np.array([{'ypix': [1]}] * 5, dtype=object))
    np.save(d / "iscell.npy", np.array([[1, 0]] * 5, dtype=object))
    np.save(d / "spks.npy", np.zeros((5, 100)))
    np.save(d / "ops.npy", np.array({'Lx': 100}, dtype=object))

    with open(d / "test.yaml", "w") as f:
        f.write("sample_rate: 30\ndata_type: 2p")

    result = binarize(str(d), verbose=False)
    assert result.sample_rate == 30
