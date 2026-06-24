import pytest
import numpy as np
from binarize2pcalcium.binarization import binarize_std

def test_binarize_std_no_events():
    traces = np.zeros((3, 100))
    out, aa = binarize_std(traces, thresh=1.0)
    assert out.shape == traces.shape
