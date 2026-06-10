"""Three-way comparison: legacy v1, legacy v2, and refactored code.

Verifies that each pipeline independently produces non-degenerate output.
The three pipelines use fundamentally different data scaling and threshold
algorithms:
  - v1: no F-scaling, detrended data for both onphase and upphase
  - v2: F/100 scaling, pre-detrend data for upphase, different max_sigma
  - refactored: no F-scaling, pre-detrend data for upphase, v1 threshold algo

What CAN be compared:
  - v1 and refactored: identical onphase and F_detrended (same algo)
  - v2 and refactored: same upphase STRATEGY (pre-detrend), different scaling
    so outputs differ, but both produce events
"""

import sys
import numpy as np
from io import StringIO

from binarize2pcalcium.pipeline import binarize
from binarize2pcalcium.data_simulation import SimulationConfig, simulate_calcium_data


def _suppress_stdout():
    return StringIO()


def _build_legacy_v1(F_raw):
    from binarize2pcalcium.binarize2pcalcium import Calcium
    c = Calcium(root_dir='.', animal_id='t', session_name='t', data_dir='.')
    c.F = F_raw.astype(np.float64)
    c.data_type = '2p'
    c.sample_rate = 30
    c.high_cutoff = 0.5
    c.percentile_threshold = 0.999
    c.dff_min = 0.05
    c.min_width_event_onphase = 30
    c.min_width_event_upphase = 10
    c.use_upphase = True
    c.remove_ends = False
    c.parallel_flag = False
    c.save_python = False
    c.save_matlab = False
    c.show_plots = False
    c.verbose = False
    c.mode_window = 900
    c.detrend_filter_threshold = 0.001
    c.detrend_model_order = 1
    return c


def _build_legacy_v2(F_raw):
    from legacy.secondversion import (
        lowpass_filter_data,
        detrend_traces as detrend_traces_v2,
        binarize_onphase as binarize_onphase_v2,
        binarize_upphase as binarize_upphase_v2,
        find_threshold,
    )
    import scipy.stats

    F = F_raw.astype(np.float64) / 100.0
    dff = F / F.mean(axis=1, keepdims=True)

    F_filt = lowpass_filter_data(dff, 0.5, 30, order=1)
    F_det = detrend_traces_v2(F_filt, 30, order=1, mode_window=900)

    thresholds = [
        find_threshold(F_det[k], 0.999, 0.05)
        for k in range(F_det.shape[0])
    ]

    moments = np.array([scipy.stats.moment(F_det[k], moment=2) for k in range(F_det.shape[0])])
    for k in range(len(thresholds)):
        if moments[k] >= 0.01:
            thresholds[k] = 0.5

    onphase = binarize_onphase_v2(F_det, thresholds, 30)
    upphase = binarize_upphase_v2(F_filt, F_det, thresholds, 10)

    return type('Result', (), dict(
        onphase=onphase, upphase=upphase, F_det=F_det, thresholds=thresholds,
    ))()


def test_v1_vs_refactored_identical():
    """v1 and refactored — same algorithm, no F-scaling → onphase matches."""
    cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
    F_noisy, _ = simulate_calcium_data(cfg)

    c = _build_legacy_v1(F_noisy)
    old = sys.stdout; sys.stdout = _suppress_stdout()
    try: c.run_binarize()
    finally: sys.stdout = old

    r3 = binarize(
        F_noisy, sample_rate=30, data_type='2p',
        high_cutoff=0.5, percentile_threshold=0.999, dff_min=0.05,
        min_width_onphase=30, min_width_upphase=10,
        remove_ends=False, parallel_flag=False,
        moment_flag=True, moment=2, moment_threshold=0.01, moment_scaling=0.5,
        mode_window=900, detrend_filter_threshold=0.001,
        verbose=False,
    )

    np.testing.assert_array_equal(r3.onphase, c.F_onphase_bin)
    np.testing.assert_array_almost_equal(r3.F_detrended, c.F_detrended, decimal=5)


def test_v2_produces_events():
    """v2 with F/100 scaling still produces onphase and upphase events."""
    cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
    F_noisy, _ = simulate_calcium_data(cfg)

    r2 = _build_legacy_v2(F_noisy)
    assert r2.onphase.sum() > 0, "v2 produced no onphase events"
    assert r2.upphase.sum() > 0, "v2 produced no upphase events"


def test_refactored_produces_events():
    """Refactored produces onphase and upphase events."""
    cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
    F_noisy, _ = simulate_calcium_data(cfg)

    r3 = binarize(
        F_noisy, sample_rate=30, data_type='2p',
        high_cutoff=0.5, percentile_threshold=0.999, dff_min=0.05,
        min_width_onphase=30, min_width_upphase=10,
        remove_ends=False, parallel_flag=False,
        moment_flag=True, mode_window=900, detrend_filter_threshold=0.001,
        verbose=False,
    )
    assert r3.onphase.sum() > 0, "refactored produced no onphase events"
    assert r3.upphase.sum() > 0, "refactored produced no upphase events"
