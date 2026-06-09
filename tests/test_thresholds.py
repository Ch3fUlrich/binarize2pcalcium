"""Tests for threshold, binarization, and detrending functions."""
import os
import numpy as np
import pytest
from binarize2pcalcium.binarize2pcalcium import (
    Calcium,
    find_threshold_by_gaussian_fit,
    find_threshold_by_gaussian_fit_parallel,
)


def test_find_threshold_by_gaussian_fit():
    """Test threshold finding from gaussian fit to fluorescence data."""
    F_filtered = np.random.randn(2, 500) * 0.1
    percentile_threshold = 0.99
    dff_min = 0.05

    thresholds = find_threshold_by_gaussian_fit(F_filtered, percentile_threshold, dff_min)
    assert len(thresholds) == 2
    assert thresholds[0] >= dff_min
    assert thresholds[1] >= dff_min


def test_find_threshold_by_gaussian_fit_noisy():
    """Test threshold with very noisy data."""
    F_filtered = np.random.randn(3, 200) * 0.5 + 0.1
    percentile_threshold = 0.999
    dff_min = 0.02

    thresholds = find_threshold_by_gaussian_fit(F_filtered, percentile_threshold, dff_min)
    assert len(thresholds) == 3
    for t in thresholds:
        assert t >= dff_min


def test_find_threshold_by_gaussian_fit_parallel():
    """Test parallel threshold finding."""
    F_detrended = np.random.randn(500) * 0.1
    cell_id = 5
    ll = [F_detrended, cell_id]
    percentile_threshold = 0.99
    snr_min = 0.05

    thresh = find_threshold_by_gaussian_fit_parallel(
        ll, percentile_threshold, snr_min, maximum_sigma=100
    )
    assert thresh >= snr_min


def test_find_threshold_by_gaussian_fit_parallel_high_max_sigma():
    """Test parallel threshold with tight sigma limit."""
    F_detrended = np.random.randn(500) * 3
    cell_id = 0
    ll = [F_detrended, cell_id]
    percentile_threshold = 0.99
    snr_min = 0.05

    thresh = find_threshold_by_gaussian_fit_parallel(
        ll, percentile_threshold, snr_min, maximum_sigma=0.01
    )
    assert thresh == 1


def test_chebyshev_filter():
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    traces = np.random.randn(2, 100)
    res = c.chebyshev_filter(traces)
    assert res.shape == traces.shape


def test_band_pass_filter():
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    traces = np.random.randn(2, 100)
    res = c.band_pass_filter(traces)
    assert res.shape == traces.shape


def test_scale_binarized():
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.min_event_amplitude = 0
    traces = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    traces_scale = np.array([[0.0, 0.0, 2.0, 5.0, 2.0, 0.5, 0.0, 0.0]])
    scaled = c.scale_binarized(traces, traces_scale)
    assert scaled.shape == traces.shape


def test_detrend_traces():
    """Test detrend_traces method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    c.detrend_model_order = 1
    c.detrend_filter_threshold = 0.001
    c.mode_window = None
    traces = np.random.randn(5, 200) * 0.1 + np.linspace(0, 1, 200)
    result = c.detrend_traces(traces)
    assert result.shape == traces.shape


def test_filter_model_order1():
    """Test filter_model with order 1 detrend."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    c.detrend_model_order = 1
    c.low_cutoff = 0.01
    c.high_cutoff = 2
    traces = np.random.randn(3, 200) * 0.1
    result = c.filter_model(traces)
    assert result.shape == traces.shape


def test_filter_model_order2():
    """Test filter_model with order 2 detrend."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    c.detrend_model_order = 2
    c.low_cutoff = 0.01
    c.high_cutoff = 2
    traces = np.random.randn(3, 200) * 0.1
    result = c.filter_model(traces)
    assert result.shape == traces.shape


def test_binarize():
    """Test the binarize method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.data_dir = '.'
    c.sample_rate = 30
    traces = np.random.randn(3, 100) * 0.5
    traces_out, traces_aa = c.binarize(traces, thresh=1.0)
    assert traces_out.shape == traces.shape


def test_binarize_derivative():
    """Test binarize_derivative method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.data_dir = '.'
    c.sample_rate = 30
    traces = np.random.randn(3, 100) * 0.5
    traces_out, traces_aa = c.binarize_derivative(traces, thresh=2)
    assert traces_out.shape == traces.shape


def test_binarize_onphase():
    """Test binarize_onphase method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    c.low_cutoff = 0.01
    c.high_cutoff = 2
    traces = np.random.randn(3, 200) * 0.3
    val_scale = np.array([0.1, 0.1, 0.1])
    result = c.binarize_onphase(traces, val_scale, min_width_event=5, min_thresh_std=1.0)
    assert result.shape == traces.shape


def test_binarize_onphase2():
    """Test binarize_onphase2 method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    c.low_cutoff = 0.01
    c.high_cutoff = 2
    c.thresholds = [0.1, 0.1, 0.1]
    traces = np.random.randn(3, 200) * 0.3
    result = c.binarize_onphase2(traces, min_width_event=5, min_thresh_std=1.0)
    assert result.shape == traces.shape


def test_smooth_traces():
    """Test smooth_traces method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    traces = np.random.randn(3, 200) * 0.1
    result = c.smooth_traces(traces, traces.copy())
    assert result.shape == traces.shape


def test_make_dir(tmp_path):
    """Test make_dir method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    test_dir = tmp_path / "test_make_dir"
    c.make_dir(str(test_dir))
    assert test_dir.exists()


def test_shuffle_rasters():
    """Test shuffle_rasters method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.sample_rate = 30
    rasters = np.random.randint(0, 2, (10, 100)).astype(float)
    rasters_dff = np.random.randn(10, 100) * 0.1
    r_out, dff_out = c.shuffle_rasters(rasters.copy(), rasters_dff.copy())
    assert r_out.shape == rasters.shape


def test_set_default_parameters_1p():
    """Test 1p parameter defaults."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.set_default_parameters_1p()
    assert c.sample_rate == 20
    assert c.inscopix_flag == False
    assert c.moment_flag == True


def test_set_default_parameters_2p():
    """Test 2p parameter defaults."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.set_default_parameters_2p()
    assert c.use_upphase == True
    assert c.parallel_flag == True


def test_compute_SNR():
    """Test compute_SNR method."""
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.F = np.random.randn(10, 100) * 0.1 + 1
    c.compute_SNR()
    assert hasattr(c, 'snrs')
    assert len(c.snrs) == 10


def test_find_threshold_by_moment(tmp_path):
    """Test find_threshold_by_moment method."""
    import matplotlib
    matplotlib.use('Agg')
    c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
    c.F_detrended = np.random.randn(8, 100) * 0.1
    c.moment = 2
    c.moment_threshold = 0.01
    c.moment_scaling = 0.5
    c.thresholds = [0.05] * 8
    c.root_dir = str(tmp_path)
    c.animal_id = 'test'
    c.session_name = 'test'
    os.makedirs(tmp_path / 'test' / 'test' / 'figures', exist_ok=True)
    c.find_threshold_by_moment()
    assert len(c.moment_values) == 8
