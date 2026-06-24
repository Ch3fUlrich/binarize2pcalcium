import pytest
import numpy as np
from binarize2pcalcium.pca import load_pca_animal

def test_load_pca_animal_missing(tmp_path):
    aucs, n_neurons = load_pca_animal(str(tmp_path), "animal1", "test", True, False)
    assert aucs is None
    assert n_neurons is None
