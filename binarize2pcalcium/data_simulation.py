"""Calcium imaging data simulation for 2P (Suite2p) and 1P (Inscopix).

Produces synthetic fluorescence traces that mimic real experimental data
and are compatible with the ``binarize()`` pipeline.

===============  ===================  ========================================
Output type      Data representation  Pipeline dF/F formula
===============  ===================  ========================================
2P (for_2p)      Photon counts        ``(F - F0) / F0``  (fractional dF/F)
1P (for_1p)      CNMF-E temporal      ``F - F0``  (baseline-subtracted,
                 components (~1-5      typical of Inscopix traces)
                 arbitrary units)
===============  ===================  ========================================

.. note::

    1P Inscopix data represents *CNMF-E-extracted temporal components*,
    not raw photon counts.  CNMF-E scales each cell's temporal trace to
    arbitrary fluorescence units (typically ~1-5).  The binarization
    pipeline subtracts the cell-specific median baseline to compute dF/F.

GCaMP6 kinetics (typical):
    - Rise time (τ_on):  0.05 - 0.15 s
    - Decay time (τ_off): 0.5 - 2.0 s

Transient model:
    f(t) = A · (exp(-Δt/τ_off) − exp(-Δt/τ_on))

Noise model (2P):
    F_obs = Poisson(F_true) + N(0, σ_read)

Noise model (1P):
    F_obs = F_true + N(0, σ_read)   (additive Gaussian, higher floor)

References
----------
    Chen et al. (2013) "Ultrasensitive fluorescent proteins for imaging
    neuronal activity."  Nature 499, 295-300.
    https://doi.org/10.1038/nature12354

    Zhou et al. (2018) "Efficient and accurate extraction of in vivo
    calcium signals from microendoscopic video data."  eLife 7, e28728.
    https://doi.org/10.7554/eLife.28728  (CNMF-E algorithm)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Transient kernel
# ---------------------------------------------------------------------------


def _transient_kernel(
    n_samples: int,
    tau_on_samp: float,
    tau_off_samp: float,
    amplitude: float,
) -> np.ndarray:
    """GCaMP transient — difference of exponentials.

    f[t] = A · (exp(−t / τ_off) − exp(−t / τ_on)),  clamped ≥ 0.
    """
    t = np.arange(n_samples, dtype=np.float64)
    tau_on = max(tau_on_samp, 1e-6)
    tau_off = max(tau_off_samp, 1e-6)
    kernel = amplitude * (np.exp(-t / tau_off) - np.exp(-t / tau_on))
    return np.maximum(kernel, 0.0)


# ---------------------------------------------------------------------------
# Slow baseline drift
# ---------------------------------------------------------------------------


def _slow_baseline_drift(n_t: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency random walk to simulate photobleaching and tissue drift."""
    steps = rng.normal(0.0, scale, size=n_t)
    drift = np.cumsum(steps)
    drift -= drift[0]
    return drift


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------


def _add_photon_noise(
    F_true: np.ndarray,
    rng: np.random.Generator,
    read_noise_std: float = 0.0,
) -> np.ndarray:
    """Poisson shot noise + Gaussian read noise."""
    F_true = np.maximum(F_true, 0.0)
    F_noisy = rng.poisson(F_true).astype(np.float64)
    if read_noise_std > 0:
        F_noisy += rng.normal(0.0, read_noise_std, size=F_noisy.shape)
    return F_noisy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Configuration for synthetic calcium imaging data."""

    # ---- Recording parameters ----
    n_cells: int = 30
    n_timepoints: int = 9000            # e.g. 5 min @ 30 Hz
    sample_rate: float = 30.0           # Hz

    # ---- GCaMP transient kinetics (seconds, per-cell mean) ----
    tau_rise_mean: float = 0.08         # GCaMP6f rise ≈ 80 ms
    tau_rise_std: float = 0.02
    tau_decay_mean: float = 0.6         # GCaMP6f decay ≈ 600 ms
    tau_decay_std: float = 0.15
    amplitude_mean: float = 0.30        # peak dF/F (30 % is realistic for GCaMP6f)
    amplitude_std: float = 0.10

    # ---- Event timing ----
    event_rate: float = 0.04            # events / second / cell (2-3 per minute)

    # ---- Baseline & noise (in photon counts) ----
    baseline_photons: float = 300.0     # mean photons / frame / cell
    baseline_std: float = 60.0          # cell-to-cell variability
    read_noise_std: float = 4.0         # Gaussian read noise (photons)
    drift_scale: float = 0.02           # slow baseline drift per frame

    # ---- Seed ----
    seed: int = 42

    # ------------------------------------------------------------------
    # Predefined configurations
    # ------------------------------------------------------------------

    @classmethod
    def for_2p(cls, **overrides) -> "SimulationConfig":
        """Preconfigured for 2-photon GCaMP6s imaging (30 Hz, wide events).

        Returns a SimulationConfig with parameters tuned so the binarization
        pipeline produces detectable onphase/upphase events on the simulated
        data.

        Accepts keyword ``**overrides`` to adjust individual parameters,
        e.g. ``SimulationConfig.for_2p(n_cells=100, seed=7)``.
        """
        params = dict(
            n_cells=30,
            n_timepoints=9000,
            sample_rate=30.0,
            tau_rise_mean=0.15,       # GCaMP6s rise
            tau_decay_mean=1.8,        # GCaMP6s decay — wide enough for detection
            amplitude_mean=0.30,
            baseline_photons=300.0,
            read_noise_std=4.0,
            event_rate=0.03,
            drift_scale=0.02,
            seed=42,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def for_1p(cls, **overrides) -> "SimulationConfig":
        """Preconfigured for 1-photon / Inscopix miniscope imaging (20 Hz).

        Simulates CNMF-E-extracted fluorescence traces (arbitrary units,
        typical range ~1-5) with realistic GCaMP kinetics, higher noise
        floor, and slow baseline drift characteristic of freely behaving
        animals.

        The raw traces are in the range expected by Inscopix preprocessing:
        CNMF-E outputs temporal components scaled to ~1-5 A.U., and the
        pipeline subtracts the median baseline (dF/F = F - F0).  Thresholds
        are tuned to detect transients of 10-30% above baseline.

        Accepts keyword ``**overrides`` to adjust individual parameters.
        """
        params = dict(
            n_cells=25,
            n_timepoints=6000,
            sample_rate=20.0,
            tau_rise_mean=0.18,
            tau_decay_mean=2.0,
            amplitude_mean=0.20,         # 20 % transient above baseline
            baseline_photons=2.0,        # CNMF-E temporal component scale (~1-5 A.U.)
            read_noise_std=0.15,         # higher relative noise for 1P
            event_rate=0.03,
            drift_scale=0.01,
            seed=42,
        )
        params.update(overrides)
        return cls(**params)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_calcium_data(
    config: SimulationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic calcium imaging fluorescence traces.

    Each cell gets its own baseline, GCaMP kinetics, event timing, and
    slow baseline drift.  The output is a noisy fluorescence matrix
    suitable as input to `binarize()`.

    For 2-photon simulations (``for_2p``), the output represents raw
    photon counts (typical range 200-400, shot-noise limited).  The
    pipeline normalizes these via (F - F0) / F0.

    For 1-photon / Inscopix simulations (``for_1p``), the output
    represents CNMF-E-extracted temporal components in arbitrary
    fluorescence units (typical range 1-5).  The pipeline subtracts the
    median baseline (F - F0) to produce dF/F, and thresholds detect
    transients of 10-30 % above baseline.

    Returns
    -------
    F_noisy : ndarray  [n_cells, n_timepoints]
        Noisy fluorescence traces.
        - 2P: photon counts (~200-400).
        - 1P: CNMF-E temporal components (~1-5 A.U.).
    dff_true : ndarray  [n_cells, n_timepoints]
        Ground-truth dF/F (fractional, noise-free, before baseline drift).
    """
    if config is None:
        config = SimulationConfig()
    cfg = config
    rng = np.random.default_rng(cfg.seed)

    n_cells = cfg.n_cells
    n_t = cfg.n_timepoints
    fs = cfg.sample_rate

    # ---- Per-cell baselines (log-normal) ----
    baselines = rng.lognormal(
        mean=np.log(cfg.baseline_photons),
        sigma=cfg.baseline_std / cfg.baseline_photons,
        size=n_cells,
    )
    baselines = np.clip(baselines, 30.0, None)

    # ---- Per-cell kinetics ----
    tau_rise = np.abs(rng.normal(cfg.tau_rise_mean, cfg.tau_rise_std, size=n_cells))
    tau_off = np.abs(rng.normal(cfg.tau_decay_mean, cfg.tau_decay_std, size=n_cells))
    amplitudes = np.abs(rng.normal(cfg.amplitude_mean, cfg.amplitude_std, size=n_cells))

    tau_rise_samp = tau_rise * fs
    tau_off_samp = tau_off * fs

    # ---- Build dF/F (noise-free transients only) ----
    dff_true = np.zeros((n_cells, n_t), dtype=np.float64)

    event_prob = cfg.event_rate / fs
    for i in range(n_cells):
        onsets = rng.random(n_t) < event_prob
        onset_idx = np.where(onsets)[0]

        for t0 in onset_idx:
            kernel_len = int(6 * tau_off_samp[i])       # ≈ 99.75 % of decay
            if t0 + kernel_len > n_t:
                kernel_len = n_t - t0
            if kernel_len < 2:
                continue
            kernel = _transient_kernel(
                kernel_len, tau_rise_samp[i], tau_off_samp[i], amplitudes[i]
            )
            dff_true[i, t0 : t0 + kernel_len] += kernel

    # ---- Add slow baseline drift (per cell, independent) ----
    F_clean = np.zeros_like(dff_true)
    for i in range(n_cells):
        drift = _slow_baseline_drift(n_t, cfg.drift_scale, rng)
        F_clean[i] = baselines[i] + drift + dff_true[i] * baselines[i]

    # ---- Add photon + read noise ----
    F_noisy = np.zeros_like(F_clean)
    for i in range(n_cells):
        F_noisy[i] = _add_photon_noise(F_clean[i], rng, cfg.read_noise_std)

    return F_noisy, dff_true


# ---------------------------------------------------------------------------
# Convenience: simulate + binarize
# ---------------------------------------------------------------------------


def simulate_and_binarize(
    config: SimulationConfig | None = None,
    binarize_kwargs: dict | None = None,
):
    """Simulate data and run the full binarization pipeline.

    Returns
    -------
    result : BinarizationResult
        Pipeline output.
    event_gt : ndarray  [n_cells, n_timepoints]
        Binary ground truth (where dF/F > 0.01).
    dff_true : ndarray  [n_cells, n_timepoints]
        Noise-free dF/F.
    """
    from binarize2pcalcium.pipeline import binarize

    F_noisy, dff_true = simulate_calcium_data(config)

    bkw = dict(sample_rate=30, data_type='2p', verbose=False)
    if binarize_kwargs is not None:
        bkw.update(binarize_kwargs)
    if config is not None:
        bkw['sample_rate'] = config.sample_rate

    result = binarize(F_noisy, **bkw)
    event_gt = (dff_true > 0.01).astype(np.uint8)

    return result, event_gt, dff_true
