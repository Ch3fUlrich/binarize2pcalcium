"""Integration tests: refactored pipeline vs legacy Calcium class.

These tests generate identical input data and run it through both the
legacy ``Calcium.run_binarize()`` and the new ``binarize()`` function,
then assert that every output array and threshold value matches exactly.
"""

import os
import sys
import numpy as np
import pytest
from io import StringIO

from binarize2pcalcium.pipeline import binarize, BinarizationResult
from binarize2pcalcium.data_simulation import SimulationConfig, simulate_calcium_data


# ── helpers ────────────────────────────────────────────────────────────────

def _suppress_stdout():
    return StringIO()


def _build_legacy_calcium(F_raw, **overrides):
    from binarize2pcalcium.binarize2pcalcium import Calcium
    c = Calcium(root_dir='.', animal_id='test', session_name='test', data_dir='.')
    c.F = F_raw.astype(np.float64)
    defaults = dict(
        data_type='2p', sample_rate=30, high_cutoff=0.5,
        detrend_model_order=1, detrend_filter_threshold=0.001, mode_window=900,
        percentile_threshold=0.999, dff_min=0.05, maximum_std_of_signal=0.08,
        min_width_event_onphase=10, min_width_event_upphase=5,
        use_upphase=True, remove_ends=False, parallel_flag=False,
        moment_flag=False, moment=2, moment_threshold=0.01, moment_scaling=0.5,
        save_python=False, save_matlab=False, show_plots=False, verbose=False,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(c, k, v)
    return c


def _new_defaults(**overrides):
    d = dict(
        sample_rate=30, data_type='2p', high_cutoff=0.5,
        percentile_threshold=0.999, dff_min=0.05,
        min_width_onphase=10, min_width_upphase=5,
        remove_ends=False, parallel_flag=False,
        moment_flag=False, verbose=False,
        mode_window=900, detrend_filter_threshold=0.001,
    )
    d.update(overrides)
    return d


# ── tests ──────────────────────────────────────────────────────────────────

class TestLegacyVsRefactored:

    def test_2p_basic_equivalence(self):
        cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
        F_noisy, _ = simulate_calcium_data(cfg)

        c = _build_legacy_calcium(F_noisy)
        old = sys.stdout; sys.stdout = _suppress_stdout()
        try: c.run_binarize()
        finally: sys.stdout = old

        result = binarize(F_noisy, **_new_defaults())

        np.testing.assert_array_equal(result.onphase, c.F_onphase_bin)
        np.testing.assert_array_equal(result.upphase, c.F_upphase_bin)
        np.testing.assert_array_almost_equal(result.F_detrended, c.F_detrended, decimal=6)
        np.testing.assert_array_almost_equal(result.F_filtered, c.F_filtered_saved, decimal=6)
        for i, (t_new, t_leg) in enumerate(zip(result.thresholds, c.thresholds)):
            np.testing.assert_almost_equal(t_new, t_leg, decimal=6)

    def test_1p_equivalence(self):
        cfg = SimulationConfig.for_1p(n_cells=5, n_timepoints=2000)
        F_noisy, _ = simulate_calcium_data(cfg)

        c = _build_legacy_calcium(F_noisy,
            data_type='1p', sample_rate=20, high_cutoff=1.0, dff_min=0.10,
            min_width_event_onphase=8, min_width_event_upphase=4, mode_window=600,
        )
        old = sys.stdout; sys.stdout = _suppress_stdout()
        try: c.run_binarize()
        finally: sys.stdout = old

        result = binarize(F_noisy, **_new_defaults(
            data_type='1p', sample_rate=20, high_cutoff=1.0, dff_min=0.10,
            min_width_onphase=8, min_width_upphase=4, mode_window=600,
        ))
        np.testing.assert_array_equal(result.onphase, c.F_onphase_bin)
        np.testing.assert_array_equal(result.upphase, c.F_upphase_bin)
        np.testing.assert_array_almost_equal(result.F_detrended, c.F_detrended, decimal=6)
        np.testing.assert_array_almost_equal(result.F_filtered, c.F_filtered_saved, decimal=6)

    def test_moment_flag_enabled(self):
        cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
        F_noisy, _ = simulate_calcium_data(cfg)

        c = _build_legacy_calcium(F_noisy,
            moment_flag=True, moment_threshold=0.005, moment_scaling=0.3)
        os.makedirs(os.path.join(c.root_dir, c.animal_id, str(c.session_name), 'figures'), exist_ok=True)
        old = sys.stdout; sys.stdout = _suppress_stdout()
        try: c.run_binarize()
        finally: sys.stdout = old

        result = binarize(F_noisy, **_new_defaults(
            moment_flag=True, moment_threshold=0.005, moment_scaling=0.3))
        for i, (t_new, t_leg) in enumerate(zip(result.thresholds, c.thresholds)):
            np.testing.assert_almost_equal(t_new, t_leg, decimal=6)

    def test_remove_ends_enabled(self):
        cfg = SimulationConfig.for_2p(n_cells=5, n_timepoints=2000)
        F_noisy, _ = simulate_calcium_data(cfg)

        c = _build_legacy_calcium(F_noisy, remove_ends=True)
        old = sys.stdout; sys.stdout = _suppress_stdout()
        try: c.run_binarize()
        finally: sys.stdout = old

        result = binarize(F_noisy, **_new_defaults(remove_ends=True))
        assert result.F_filtered.shape == c.F_filtered_saved.shape
        assert result.F_detrended.shape == c.F_detrended.shape

    def test_maximum_std_of_signal(self):
        """Parallel path must be used for maximum_std_of_signal to take effect."""
        F_noisy = np.random.randn(32, 500) * 3.0 + 10.0
        result = binarize(F_noisy, **_new_defaults(
            parallel_flag=True, maximum_std_of_signal=0.001,
        ))
        for t in result.thresholds:
            np.testing.assert_almost_equal(t, 1.0, decimal=3)

    def test_parallel_sequential_identical(self):
        cfg = SimulationConfig.for_2p(n_cells=8, n_timepoints=2000)
        F_noisy, _ = simulate_calcium_data(cfg)
        r_par = binarize(F_noisy, **_new_defaults(parallel_flag=True))
        r_seq = binarize(F_noisy, **_new_defaults(parallel_flag=False))
        np.testing.assert_array_equal(r_par.onphase, r_seq.onphase)
        np.testing.assert_array_equal(r_par.upphase, r_seq.upphase)


class TestBinarizationResultSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = SimulationConfig.for_2p(n_cells=3, n_timepoints=500)
        F_noisy, _ = simulate_calcium_data(cfg)
        result = binarize(F_noisy, **_new_defaults())
        path = tmp_path / "test.npz"
        result.save(str(path))
        loaded = BinarizationResult.load(str(path))
        np.testing.assert_array_equal(result.onphase, loaded.onphase)
        np.testing.assert_array_equal(result.upphase, loaded.upphase)


class TestEdgeCases:
    def test_no_upphase_flag(self):
        cfg = SimulationConfig.for_2p(n_cells=3, n_timepoints=500)
        F_noisy, _ = simulate_calcium_data(cfg)
        result = binarize(F_noisy, **_new_defaults(use_upphase=False))
        assert np.all(result.upphase == 0)

    def test_single_cell(self):
        F = np.random.randn(1, 500) * 0.1 + 5.0
        result = binarize(F, **_new_defaults())
        assert result.onphase.shape == (1, 500)
