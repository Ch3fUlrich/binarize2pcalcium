import pytest
import numpy as np
import os
from binarize2pcalcium.binarize2pcalcium import Calcium

def test_legacy_init():
    c = Calcium('.', 'test_animal', 'test_session')
    assert c.root_dir == '.'

def test_legacy_set_params():
    c = Calcium('.', 'test_animal', 'test_session')
    c.sample_rate = 30
    assert c.sample_rate == 30

def test_legacy_methods(tmp_path):
    import sys
    from io import StringIO
    c = Calcium(str(tmp_path), 'test_animal', 'test_session', str(tmp_path))
    c.F = np.random.randn(5, 500) * 0.1 + 0.5
    c.data_type = '2p'

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        c.run_binarize()
    finally:
        sys.stdout = old_stdout

    assert c.F_onphase_bin is not None
    assert c.F_upphase_bin is not None
