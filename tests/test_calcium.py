"""Tests for the Calcium class core methods."""
import os
import numpy as np
import pytest
from binarize2pcalcium.binarize2pcalcium import Calcium


class TestCalciumInit:
    """Tests for Calcium class initialization."""

    def test_init_with_all_params(self, tmp_path):
        """Test initialization with all parameters."""
        data_dir = tmp_path / "2pdata"
        animal_id = "DON-test"
        session_name = "test_session"
        os.makedirs(data_dir / animal_id / session_name / "suite2p" / "plane0", exist_ok=True)

        c = Calcium(
            root_dir=str(data_dir),
            animal_id=animal_id,
            session_name=session_name,
            data_dir=str(data_dir),
        )
        assert c.animal_id == animal_id
        assert c.root_dir == str(data_dir)
        assert c.session_name == session_name
        assert c.data_dir == str(data_dir)

    def test_init_defaults(self):
        """Test initialization with empty/default parameters."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        assert c.verbose == False
        assert c.remove_bad_cells == True
        assert c.recompute == False
        assert c.n_cores == 16
        assert c.check_zero_cells == True
        assert c.save_figures == True


class TestCalciumStandardize:
    """Tests for standardize method."""

    def test_standardize_shape(self):
        """Test standardize preserves shape."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.data_dir = '.'
        traces = np.random.randn(10, 100)
        std_traces = c.standardize(traces)
        assert std_traces.shape == traces.shape

    def test_standardize_range(self):
        """Test standardized data has range [0, 1]."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.data_dir = '.'
        traces = np.random.randn(5, 200) * 2 + 3
        std_traces = c.standardize(traces)
        for k in range(traces.shape[0]):
            assert np.isclose(np.max(std_traces[k]) - np.min(std_traces[k]), 1.0, atol=0.01)


class TestCalciumFiltering:
    """Tests for filtering methods."""

    def test_high_pass_filter(self):
        """Test high_pass_filter preserves shape."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.low_cutoff = 0.01
        traces = np.random.randn(10, 100)
        filtered = c.high_pass_filter(traces)
        assert filtered.shape == traces.shape

    def test_low_pass_filter(self):
        """Test low_pass_filter preserves shape."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.high_cutoff = 2
        traces = np.random.randn(10, 100)
        filtered = c.low_pass_filter(traces)
        assert filtered.shape == traces.shape

    def test_detrend(self):
        """Test detrend preserves shape."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        traces = np.random.randn(10, 100)
        detrended = c.detrend(traces)
        assert detrended.shape == traces.shape

    def test_medfilt(self):
        """Test median filter on 1D array."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        trace = np.random.randn(100)
        med_filtered = c.medfilt(trace, 5)
        assert len(med_filtered) == len(trace)

    def test_medfilt_kernel_validation(self):
        """Test medfilt enforces odd kernel size."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        with pytest.raises(AssertionError, match="Median filter length must be odd"):
            c.medfilt(np.random.randn(100), 4)

    def test_medfilt_dim_validation(self):
        """Test medfilt enforces 1D input."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        with pytest.raises(AssertionError, match="Input must be one-dimensional"):
            c.medfilt(np.random.randn(10, 100), 5)

    def test_wavelet_filter(self):
        """Test wavelet_filter preserves shape."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        traces = np.random.randn(1, 100)
        filtered = c.wavelet_filter(traces)
        assert filtered.shape == traces.shape

    def test_chebyshev_filter(self):
        """Test chebyshev bandpass filter."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.low_cutoff = 0.5
        c.high_cutoff = 2
        traces = np.random.randn(3, 100)
        result = c.chebyshev_filter(traces)
        assert result.shape == traces.shape

    def test_band_pass_filter(self):
        """Test band_pass_filter."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.low_cutoff = 0.5
        c.high_cutoff = 2
        traces = np.random.randn(3, 100)
        result = c.band_pass_filter(traces)
        assert result.shape == traces.shape


class TestCalciumBinarization:
    """Tests for binarization methods."""

    def test_binarize(self):
        """Test binarize method."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.data_dir = '.'
        c.sample_rate = 30
        traces = np.random.randn(3, 100)
        out, aa = c.binarize(traces, thresh=1.0)
        assert out.shape == traces.shape
        assert aa.shape == traces.shape

    def test_binarize_derivative(self):
        """Test binarize_derivative."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.data_dir = '.'
        c.sample_rate = 30
        traces = np.random.randn(3, 100)
        out, aa = c.binarize_derivative(traces, thresh=2)
        assert out.shape == traces.shape

    def test_binarize_onphase(self):
        """Test binarize_onphase."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.low_cutoff = 0.01
        c.high_cutoff = 2
        traces = np.random.randn(3, 200) * 0.3
        val_scale = np.array([0.1, 0.1, 0.1])
        result = c.binarize_onphase(traces, val_scale, min_width_event=5, min_thresh_std=1.0)
        assert result.shape == traces.shape

    def test_binarize_onphase2(self):
        """Test binarize_onphase2 with thresholds."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        c.low_cutoff = 0.01
        c.high_cutoff = 2
        c.thresholds = [0.1, 0.1, 0.1, 0.1, 0.1]
        traces = np.random.randn(5, 200) * 0.3
        result = c.binarize_onphase2(traces, min_width_event=5, min_thresh_std=1.0)
        assert result.shape == traces.shape

    def test_scale_binarized(self):
        """Test scale_binarized."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.min_event_amplitude = 0
        traces = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        traces_scale = np.array([[0.0, 0.0, 2.0, 5.0, 2.0, 0.5, 0.0, 0.0]])
        scaled = c.scale_binarized(traces, traces_scale)
        assert scaled.shape == traces.shape


class TestCalciumUtilities:
    """Tests for utility methods."""

    def test_smooth_traces(self):
        """Test smooth_traces."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        traces = np.random.randn(5, 200) * 0.1
        result = c.smooth_traces(traces, traces.copy())
        assert result.shape == traces.shape

    def test_shuffle_rasters(self):
        """Test shuffle_rasters."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.sample_rate = 30
        rasters = np.random.randint(0, 2, (8, 100)).astype(float)
        dff = np.random.randn(8, 100) * 0.1
        r_out, dff_out = c.shuffle_rasters(rasters.copy(), dff.copy())
        assert r_out.shape == rasters.shape
        assert dff_out.shape == dff.shape

    def test_make_dir(self, tmp_path):
        """Test make_dir creates directory."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        test_dir = tmp_path / "new_dir"
        c.make_dir(str(test_dir))
        assert test_dir.exists()

    def test_make_dir_existing(self, tmp_path):
        """Test make_dir with existing directory doesn't fail."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        test_dir = tmp_path / "existing"
        os.makedirs(test_dir)
        c.make_dir(str(test_dir))  # Should not raise
        assert test_dir.exists()


class TestCalciumParameters:
    """Tests for parameter setting methods."""

    def test_set_default_parameters_1p(self):
        """Test 1p parameters are set correctly."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.set_default_parameters_1p()
        assert c.sample_rate == 20
        assert c.inscopix_flag == False
        assert c.moment_flag == True

    def test_set_default_parameters_2p(self):
        """Test 2p parameters are set correctly."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.set_default_parameters_2p()
        assert c.use_upphase == True
        assert c.parallel_flag == True

    def test_set_default_parameters_1p_save_flags(self):
        """Test 1p save flags."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.set_default_parameters_1p()
        assert c.save_python == True
        assert c.save_matlab == True


class TestCalciumSNR:
    """Tests for SNR computation."""

    def test_compute_SNR(self):
        """Test compute_SNR sets snrs and skews attributes."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.F = np.random.randn(15, 100) * 0.1 + 1
        c.compute_SNR()
        assert hasattr(c, 'snrs')
        assert len(c.snrs) == 15
        assert hasattr(c, 'skews')
        assert len(c.skews) == 15

    def test_compute_SNR_small(self):
        """Test compute_SNR with minimal data."""
        c = Calcium(root_dir='', animal_id='', session_name='', data_dir='')
        c.F = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        c.compute_SNR()
        assert len(c.snrs) == 1
        assert len(c.skews) == 1


class TestCalciumMoment:
    """Tests for moment-based threshold."""

    def test_find_threshold_by_moment(self, tmp_path):
        """Test find_threshold_by_moment."""
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
        # Create figures directory to prevent FileNotFoundError
        os.makedirs(tmp_path / 'test' / 'test' / 'figures', exist_ok=True)
        c.find_threshold_by_moment()
        assert len(c.moment_values) == 8
