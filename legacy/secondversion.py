import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, resample_poly
import scipy
from scipy import stats
from tqdm import tqdm
from statistics import NormalDist
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return b, a


def lowpass_filter_data(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data, axis=1)


def detrend_traces(traces, sample_rate, order=1, mode_window=None):
    out = traces.copy()
    n_samples = traces.shape[1]
    t = np.arange(n_samples)

    for k in tqdm(range(traces.shape[0]), desc='detrending'):
        temp = traces[k].copy()
        trend = lowpass_filter_data(temp[np.newaxis, :], 0.001, sample_rate, order=1)[0]

        if order >= 1:
            z = np.polyfit(t, trend, order)
            temp = temp - np.poly1d(z)(t)

        if mode_window is None:
            y, edges = np.histogram(temp, bins=np.arange(-1, 1, 0.001))
            temp -= edges[np.argmax(y)]
        else:
            for q in range(0, n_samples, mode_window):
                seg = temp[q:q+mode_window]
                y, edges = np.histogram(seg, bins=np.arange(-5, 5, 0.001))
                temp[q:q+mode_window] = seg - edges[np.argmax(y)]

        out[k] = temp
    return out


def find_threshold(cell_trace, percentile=0.9999, dff_min=0.03, max_sigma=0.8):
    try:
        y, edges = np.histogram(cell_trace, bins=np.arange(-25, 25, 0.001))
        mode_val = edges[np.argmax(y)]
        neg = cell_trace[cell_trace <= mode_val]
        pooled = np.hstack((neg, -neg.copy()))

        norm = NormalDist.from_samples(pooled)
        x = np.arange(-25, 25, 0.001)
        pdf = stats.norm.pdf(x, norm.mean, norm.stdev)
        pdf /= np.max(pdf)
        cumsum = np.cumsum(pdf)
        cumsum /= np.max(cumsum)

        idx = np.where(cumsum > percentile)[0]
        thresh = x[idx[0]]
        if norm.stdev > max_sigma:
            thresh = 1.0
        return max(thresh, dff_min)
    except Exception:
        return dff_min


def binarize_onphase(traces, thresholds, min_width):
    binary = np.zeros_like(traces)
    for k in tqdm(range(traces.shape[0]), desc='binarizing onphase'):
        temp = (traces[k] >= thresholds[k]).astype(np.float32)
        if temp.sum() == 0:
            continue
        peaks, _ = find_peaks(temp)
        if len(peaks) == 0:
            continue
        widths, _, starts, ends = peak_widths(temp, peaks)
        valid = widths >= min_width
        if valid.sum() == 0:
            continue
        for s, e in zip(np.int32(starts[valid]), np.int32(ends[valid])):
            binary[k, s:e] = 1
    return binary


def binarize_upphase(traces, traces_grad, thresholds, min_width):
    binary = np.zeros_like(traces)
    der = np.gradient(traces_grad, axis=1)
    for k in tqdm(range(traces.shape[0]), desc='binarizing upphase'):
        temp = traces[k].copy()
        temp[der[k] <= 0] = 0
        temp = (temp >= thresholds[k]).astype(np.float32)
        if temp.sum() == 0:
            continue
        peaks, _ = find_peaks(temp)
        if len(peaks) == 0:
            continue
        widths, _, starts, ends = peak_widths(temp, peaks)
        valid = widths >= min_width
        if valid.sum() == 0:
            continue
        for s, e in zip(np.int32(starts[valid]), np.int32(ends[valid])):
            binary[k, s:e] = 1
    return binary


def binarize_folder(folder_path,
                    output_dir=None,
                    output_name=None,
                    raster_dir=None,
                    overwrite=False,
                    sample_rate=20,
                    output_rate=None,
                    high_cutoff=0.5,
                    detrend_order=1,
                    mode_window=900,
                    percentile_threshold=0.9999,
                    dff_min=0.03,
                    max_std_signal=0.8,
                    moment_threshold = 0.01,   # same default as Catalino
                    moment_scaling   = 0.5, 
                    min_width_onphase=None,
                    min_width_upphase=None,
                    save_npz=True,
                    save_plots=True):
    """
    fucking catalino 
    """
    folder_path = os.path.abspath(folder_path)

    # ---------- LOAD ----------
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    f_npy = os.path.join(folder_path, 'F.npy')
    iscell_npy = os.path.join(folder_path, 'iscell.npy')

    if csv_files:
        df = pd.read_csv(csv_files[0], skiprows=2, header=None)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        if df.isna().any().any():
            df = df.ffill().bfill()
        data = df.values.astype(np.float32)
        F = data[:, 1:].T
        dtype = '1p'
    elif os.path.exists(f_npy):
        F = np.load(f_npy)
        if os.path.exists(iscell_npy):
            iscell = np.load(iscell_npy)
            good = np.where(iscell[:, 0] == 1)[0]
            F = F[good]
        dtype = '2p'
        print("in2pmotherfucker")
    else:
        raise FileNotFoundError(f"No .csv or F.npy found in {folder_path}")

    F = F/100
    n_cells, n_times = F.shape

    # ---------- OUTPUT PATHS ----------
    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if output_name is None:
            output_name = os.path.basename(folder_path)
        out_base = os.path.join(output_dir, output_name)
    else:
        out_base = os.path.join(folder_path, 'binarized')

    fname_npz = out_base + '_binarized.npz'

    # ---------- RASTER DIR ----------
    if raster_dir is None and output_dir is not None:
        raster_dir = os.path.join(os.path.dirname(output_dir), 'rasterplots')
    if raster_dir is not None:
        raster_dir = os.path.abspath(raster_dir)
        os.makedirs(raster_dir, exist_ok=True)

    # ---------- SKIP CHECK ----------
    if os.path.exists(fname_npz) and not overwrite:
        print(f"SKIP: {fname_npz} already exists.")
        return None

    # ---------- PREPROCESS ----------
    if min_width_onphase is None:
        min_width_onphase = sample_rate
    if min_width_upphase is None:
        min_width_upphase = sample_rate // 3

    f0 = np.abs(np.median(F, axis=1, keepdims=True))
    dff = (F - f0) / f0 if dtype == '2p' else F - f0

    print(f"Processing: {folder_path}  ({n_cells} cells x {n_times} frames @ {sample_rate}Hz)")
    F_filt = lowpass_filter_data(dff, high_cutoff, sample_rate, order=1)
    F_det = detrend_traces(F_filt, sample_rate, order=detrend_order, mode_window=mode_window)

    # ---------- THRESHOLDS ----------
    thresholds = np.array([
        find_threshold(F_det[k], percentile_threshold, dff_min, max_std_signal)
        for k in range(n_cells)
    ])
    
    # ---------- MOMENT CORRECTION ----------
    moment_values = np.zeros(n_cells)
    for k in range(n_cells):
        moment_values[k] = scipy.stats.moment(F_det[k], moment=2)
        if moment_values[k] >= moment_threshold:
            thresholds[k] = moment_scaling

    # ---------- BINARIZE ----------
    F_on = binarize_onphase(F_det, thresholds, min_width_onphase)
    F_up = binarize_upphase(F_filt, F_det, thresholds, min_width_upphase)

    # ---------- RESAMPLE just in case ----------
    effective_rate = sample_rate
    print(output_rate, sample_rate)
    if output_rate is not None and output_rate != sample_rate:
        print(f"Resampling {sample_rate}Hz -> {output_rate}Hz")
        F      = resample_poly(F,      up=output_rate, down=sample_rate, axis=1)
        dff    = resample_poly(dff,    up=output_rate, down=sample_rate, axis=1)
        F_det  = resample_poly(F_det,  up=output_rate, down=sample_rate, axis=1)
        F_on   = resample_poly(F_on,   up=output_rate, down=sample_rate, axis=1)
        F_on   = np.round(F_on).astype(np.float32)
        F_up   = resample_poly(F_up,   up=output_rate, down=sample_rate, axis=1)
        F_up   = np.round(F_up).astype(np.float32)
        effective_rate = output_rate

    # ---------- PACKAGE ----------
    print(len(F))
    result = {
        'F_raw': F,
        'DFF': dff,
        'F_detrended': F_det,
        'F_onphase': F_on,
        'F_upphase': F_up,
        'thresholds': thresholds,
        'sample_rate': effective_rate,
        'high_cutoff': high_cutoff,
    }

    # ---------- SAVE NPZ ----------
    if save_npz:
        np.savez_compressed(fname_npz, **result)
        print(f"Saved: {fname_npz}")

    # ---------- PLOTS ----------
    if save_plots:
        base_name = os.path.basename(out_base)
        n_times_out = F_up.shape[1]

        # RASTER
        if raster_dir is not None:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(F_up, aspect='auto', cmap='Greys', interpolation='none',
                      extent=[0, n_times_out/effective_rate, n_cells, 0])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neuron')
            ax.set_title(f'Upphase: {base_name} ({effective_rate}Hz)')
            plt.tight_layout()
            plt.savefig(os.path.join(raster_dir, f'{base_name}_raster.png'), dpi=200)
            plt.close()
            print(f"Raster saved to {os.path.join(raster_dir, base_name + '_raster.png')}")

        # SAMPLE TRACES
        fig_dir = os.path.join(os.path.dirname(out_base), 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        idx = np.random.choice(n_cells, min(20, n_cells), replace=False)
        fig, ax = plt.subplots(figsize=(20, 10))
        t = np.arange(n_times_out) / effective_rate
        for i, cid in enumerate(idx):
            off = i * 3
            ax.plot(t, F_det[cid] + off, 'b', lw=0.5)
            ax.fill_between(t, off, F_on[cid]*2 + off, color='orange', alpha=0.3)
            ax.fill_between(t, off, F_up[cid]*2 + off, color='green', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron')
        ax.set_title(f'Traces: {base_name} ({effective_rate}Hz)')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{base_name}_sample_traces.png'), dpi=200)
        plt.close()

    return result


def plot_raster(data):
    plt.figure(figsize=(12, 8))
    
    for neuron_idx in range(data.shape[0]):
        spike_times = np.where(data[neuron_idx, :] == 1)[0]
        if len(spike_times) > 0:
            plt.scatter(spike_times, 
                       np.ones_like(spike_times) * neuron_idx, 
                       s=1, c='black', marker='|')
    
    plt.xlabel('Time (bins)')
    plt.ylabel('Neuron ID')
    plt.title('Raster Plot')
    plt.ylim(-0.5, data.shape[0] - 0.5)
    plt.gca().invert_yaxis()  # Invert y-axis so neuron 1 is at the top
    plt.tight_layout()
    plt.show()