import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
from tqdm import tqdm
from tqdm import trange
import pickle
from joblib import Parallel, delayed
import concurrent.futures

import mne

#mne.set_log_level('WARNING')
mne.set_log_level('ERROR')  # quiet mne messages

from mne_connectivity import phase_slope_index
from mne_connectivity.viz import plot_connectivity_circle
#from mne_connectivity.viz import plot_connectivity_circle

from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert

from scripts.eeg.preprocess import compute_csd
from scripts.eeg.preprocess_run import get_subject_ids
from scripts.eeg.connectivity_plot import plot_save_psi_matrix, plot_save_connectivity_circle, save_psi_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mne_connectivity.viz import plot_connectivity_circle


BAND_DEFS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def create_epochs(raw, epoch_duration):
    """
    Create epochs from raw data with fixed-length events.
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        epoch_duration (float): Duration of each epoch in seconds.
        
    Returns:
        mne.Epochs: Epochs created from the raw data.
    """
    # Create fixed-length events
    events = mne.make_fixed_length_events(raw, id=1, duration=epoch_duration)
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=epoch_duration,
                        baseline=None, preload=True, reject_by_annotation=False)
    
    return epochs

def get_phase(epochs, method='wavelet', f=(0.5, 4)):
    """
    Extract phase information from EEG epochs using Hilbert or wavelet transform.

    Parameters:
        epochs (mne.Epochs): Preprocessed EEG data.
        method (str): 'hilbert' or 'wavelet'.
        f (tuple): Frequency band (e.g., (0.5, 4) for delta).

    Returns:
        phase (ndarray): Phase data, shape (n_channels, n_times).
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']

    if method == 'hilbert':
        # --- Hilbert Transform ---
        analytic_signal = hilbert(data, axis=-1)
        phase = np.angle(analytic_signal)
        #phase = np.mean(phase, axis=0)  # average over epochs → (n_channels, n_times)

    elif method == 'wavelet':
        # --- Wavelet Transform ---
        freqs = np.linspace(f[0], f[1], 5)
        phase = tfr_array_morlet(
            data, sfreq=sfreq, freqs=freqs,
            n_cycles=freqs / 2, output='phase'
        )
        # average over epochs and frequencies
        phase = np.mean(phase, axis=(0, 2))  # (n_channels, n_times)

    else:
        raise ValueError("method must be 'hilbert' or 'wavelet'")

    return phase

def proportional_thresholding(psi_matrix, proportion):
    """
    Keep only the top 'proportion' strongest connections in absolute PSI value.

    Parameters
    ----------
    psi_matrix : ndarray
        (n_channels, n_channels) connectivity matrix (symmetric).
    proportion : float
        Fraction of edges to keep (0 < proportion <= 1).

    Returns
    -------
    adj : ndarray
        Thresholded adjacency matrix.
    """

    n = psi_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    values = np.abs(psi_matrix[triu_idx])

    cutoff = np.quantile(values, 1 - proportion)

    adj = np.where(np.abs(psi_matrix) >= cutoff, psi_matrix, 0.0)

    return adj

def apply_thresholding(psi_matrix, threshold_type="thresholded", network_type="weighted",
                       threshold=None, proportion=None):
    """ 
    
    """
    adj = np.copy(psi_matrix)

    
    if threshold_type == "proportional":
        if proportion is None:
            raise ValueError("Proportional mode requires 'proportion' argument.")
        adj = proportional_thresholding(adj, proportion)

    elif threshold_type == "thresholded":
        if threshold is None:
            raise ValueError("Thresholded mode requires 'threshold' argument.")
        adj = np.where(np.abs(adj) >= threshold, adj, 0.0)

    else:
        raise ValueError("threshold_type must be 'thresholded' or 'proportional'.")

    
    if network_type == "binary":
        adj = np.where(adj != 0, 1.0, 0.0)
    elif network_type == "weighted":
        pass 
    else:
        raise ValueError("network_type must be 'binary' or 'weighted'.")

    return adj

def analyze_network_from_psi(adj, mode="weighted"):
    """
    Analyze functional connectivity network (PSI/PLI/PLV matrix) 
    using either weighted or binary approach.

    Parameters
    ----------
    psi_matrix : ndarray
        Shape (n_channels, n_channels), adjacency matrix of PSI/PLI/PLV.
    threshold : float or None, optional
        If set, values below this are removed (set to 0). Used only if mode="binary".
        Example: threshold=0.3 keeps edges with PSI >= 0.3.
    mode : {"weighted", "binary"}, default="weighted"
        Whether to perform weighted or binary network analysis.

    Returns
    -------
    results : dict
        Graph-theoretical metrics (centralities, efficiencies, etc.).
    """
  
    G = nx.from_numpy_array(adj)

    results = {}
    results["mode"] = mode
    results["density"] = np.count_nonzero(adj) / adj.size

    # --- Centrality measures ---
    results["degree_centrality"] = nx.degree_centrality(G)
    if mode == "weighted":
        results["betweenness_centrality"] = nx.betweenness_centrality(G, weight='weight')
        results["eigenvector_centrality"] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        results["node_strength"] = dict(G.degree(weight='weight'))
    else:  # binary
        results["betweenness_centrality"] = nx.betweenness_centrality(G)
        results["eigenvector_centrality"] = nx.eigenvector_centrality(G, max_iter=1000)
        results["node_strength"] = dict(G.degree())

    # --- Efficiency ---
    results["global_efficiency"] = nx.global_efficiency(G)
    results["local_efficiency"] = nx.local_efficiency(G)

    return results

def save_results_to_txt(results, id, condition, output_path, filename=None):

    subject_dir = os.path.join(output_path, id)

    # Create the directory if it doesn't exist
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    
    # Save the matrix as a numpy file
    if filename == None:
        filename = os.path.join(subject_dir, f'network_analysis_results_{id}_{condition}.txt')
    else:
        filename = os.path.join(subject_dir, f'{filename}_{id}_{condition}.txt')
    
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"=== {key} ===\n")

            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"{k}: {v}\n")
            else:
                # If value is a single number or array, convert to string
                f.write(f"{value}\n")
            f.write("\n")

def save_significant_psi(psi_matrix, subject_id, condition, output_path):
    """
    Save nonzero elements from the thresholded PSI matrix to a text file.
    Diagonal elements (r == c) are excluded.

    Args:
        psi_matrix (np.ndarray): Thresholded PSI matrix (2D NumPy array).
        subject_id (str): Subject identifier.
        condition (str): Experimental condition label.
        output_path (str): Directory where the results will be saved.

    Returns:
        int: Number of elements saved.
    """
    # Get indices of nonzero elements
    rows, cols = np.nonzero(psi_matrix)

    # Exclude diagonal elements (r == c)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]

    subject_dir = os.path.join(output_path, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # Define output file path
    filename = os.path.join(subject_dir, f'psi_output_{subject_id}_{condition}.txt')

    # Write results
    with open(filename, "w", encoding="utf-8") as f:
        f.write("List of nonzero PSI values (after thresholding)\n")
        f.write(f"Total count: {len(rows)}\n\n")
        for r, c in zip(rows, cols):
            value = psi_matrix[r, c]
            f.write(f"Index: ({r}, {c}), Value: {value:.5f}\n")

    print(f"Results saved to '{filename}'.")
    return len(rows)

def compute_phase_phase_matrix(args, phase, id, condition):
    """
    Compute n:m Phase Synchronization Index between all pairs of EEG channels.

    Parameters:
    -----------
    args : argparse.Namespace
        Contains paths and parameters.
    phase : np.ndarray, shape (n_channels, n_times)
        Phase information for the EEG channels.
    id : str
        Subject ID (e.g., 'sub-010002').
    condition : str
        Condition label (e.g., 'EO', 'EC').
    --------
    phase_phase_matrix : np.ndarray, shape (n_channels, n_channels)
        Matrix of n:m Phase Synchronization Index (PSI).
    """

    # Initialize the phase-phase matrix
    n_channels = phase.shape[0]
    psi_matrix = np.zeros((n_channels, n_channels))

    # Compute n:m phase-phase coupling for each channel pair
    for i in tqdm(range(n_channels), desc=f"Computing PSI ({id}, {condition})", leave=True):
        for j in range(n_channels):
            phase_diff = phase[j] - phase[i]
            # Compute PSI for the current m
            psi = np.abs(np.mean(np.exp(1j * phase_diff)))
            psi_matrix[i, j] = psi

    psi_matrix = np.array(psi_matrix)
    #psi_matrix = np.mean(psi_matrix, axis=0)

    # Save the matrix to a file
    save_psi_matrix(psi_matrix, id, condition, args.psi_path)

    return psi_matrix

def get_subject_metadata(subject_id, metadata_path):
    """ Retrieve metadata for a given subject from the metadata DataFrame.
    Parameters:
    - subject_id : str
        Subject identifier.
    - meta_df : pd.DataFrame
        DataFrame containing metadata with columns 'subject_id', '
    """
    meta_df = pd.read_csv(metadata_path)
    meta_df.columns = ["subject_id", "gender_code", "age_range"]

    row = meta_df[meta_df["subject_id"] == subject_id]
    if row.empty:
        return {"sex": "unknown", "group": "unknown"}

    gender_code = int(row.iloc[0]["gender_code"])
    if gender_code == 1:
        sex = "female"
    elif gender_code == 2:
        sex = "male"
    else:
        sex = "unknown"

    age_range = str(row.iloc[0]["age_range"])
    if any(x in age_range for x in ["20", "25", "30"]):
        group = "young"
    elif any(x in age_range for x in ["65", "70", "75", "80"]):
        group = "old"
    else:
        group = "unknown"

    return {"sex": sex, "group": group}

def save_global_metrics_csv(metrics_dict, subject_id, sex, group, condition, output_path, file_name):
    """ Save global network metrics to a CSV file.
    Parameters:
    - metrics_dict : dict
        Dictionary containing global network metrics (e.g., global efficiency).
    - subject_id : str
        Subject identifier.
    - group : str
        Group label (e.g., 'young', 'elderly').
    - condition : str
        Condition label (e.g., 'EO', 'EC').
    - output_path : str
        Directory to save the CSV file.
    """

    csv_path = os.path.join(output_path, file_name)

    row = {
        "subject_id": subject_id,
        "sex":sex,
        "group": group,
        "condition": condition,
        
    }
    row.update(metrics_dict)

    df = pd.DataFrame([row])
    print(df)
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

    print(f"Saved metrics for {subject_id} ({sex}, {group}, {condition}) to {csv_path}")

def compute_psi_matrix(args, raw, id, condition, method='hilbert'):
    """
    Compute Phase Synchronization Index (PSI) matrix for a given subject and condition.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Contains paths and parameters.
    raw : mne.io.Raw
        Raw EEG data (preprocessed).
    id : str
        Subject ID (e.g., 'sub-010002').
    condition : str
        Condition label (e.g., 'EO', 'EC').
    method : str
        'hilbert' or 'wavelet'
    
    Returns:
    --------
    psi_matrix : np.ndarray, shape (n_channels, n_channels)
        Matrix of Phase Synchronization Index (PSI).
    """

    # Pick only EEG channels before computing CSD

    raw = raw.copy().pick("eeg")

    # Compute CSD for both bands
    raw = compute_csd(raw)
    
    epochs = create_epochs(raw, args.epoch_duration)
    #print(f"{id}_{condition} drop log:\n", epochs.drop_log)

    if len(epochs) == 0:
        raise ValueError("No valid epochs found. Please check epoch rejection or data loading.")
    #data = epochs.get_data()
    #print(data.shape)  # shape: (n_epochs, n_channels, n_times)

    phase = get_phase(epochs, method=method)

    psi_matrix = compute_phase_phase_matrix(args, phase, id, condition)
    plot_save_psi_matrix(psi_matrix, epochs.ch_names, id, condition, args.psi_path)

    #adj1 = apply_thresholding(psi_matrix, threshold_type="thresholded", network_type="weighted", threshold=0.8)
    #adj2 = apply_thresholding(psi_matrix, threshold_type="thresholded", network_type="binary", threshold=0.8)
    adj3 = apply_thresholding(psi_matrix, threshold_type="proportional", network_type="weighted", proportion=0.2)
    #adj4 = apply_thresholding(psi_matrix, threshold_type="proportional", network_type="binary", proportion=0.2)

    count = save_significant_psi(adj3, id ,condition, args.connectivity_path)
    print(f"{count} significant PSI values have been saved.")
    
    plot_save_connectivity_circle(adj3, epochs.ch_names, id, condition, args.connectivity_path)

    adj3_results = analyze_network_from_psi(adj3, mode="weighted")
    save_results_to_txt(adj3_results, id, condition, args.connectivity_path)

    results = analyze_network_from_psi(psi_matrix, mode="weighted")
    save_results_to_txt(results, id, condition, args.connectivity_path, "all_result_analyze_network")

    # Load metadata for subject
    meta_info = get_subject_metadata(id, args.metadata_path)
    sex = meta_info["sex"]
    group = meta_info["group"]

    save_global_metrics_csv({
        "global_efficiency": adj3_results['global_efficiency'],
        "local_efficiency": adj3_results['local_efficiency'],
        "density": adj3_results['density']
    }, subject_id=id, sex=sex, group=group, condition=condition, output_path=args.connectivity_path, file_name='global_network_metrics.csv')

    save_global_metrics_csv({
        "global_efficiency": results['global_efficiency'],
        "local_efficiency": results['local_efficiency'],
        "density": results['density']
    }, subject_id=id, sex=sex, group=group, condition=condition, output_path=args.connectivity_path, file_name='all_global_network_metrics.csv')

    
    
    return psi_matrix


def load_psi_matrix(args, id, condition):
    """ Load the saved n:m Phase Synchronization Index (PSI) matrix from a file.
    Returns:
        psi_matrix (ndarray): Loaded n:m Phase Synchronization Index matrix.
    """
    path = os.path.join(args.psi_path, id, f"{id}_{condition}_psi_matrix.npy")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    loaded_matrix = np.load(path, allow_pickle=True)

    return loaded_matrix

def run_psi(args):

    subject_ids = get_subject_ids(args.preprocess_path)    

    for id in tqdm(subject_ids, desc="Connectivity Subjects"):
        for condition in ['EO', 'EC']:
            print(f"==={id}_{condition}===")
            # Define the paths to the raw EEG files
            path = os.path.join(args.preprocess_path, id, f"{id}_{condition}_eeg.fif")

            # Load the raw data
            if not os.path.exists(path):
                print(f"Raw EEG file for condition {condition} does not exist: {path}")
                continue
            # Load the raw data   
            raw = mne.io.read_raw_fif(path, preload=True)
            if not raw.preload:
                raw.load_data()

            psi_matrix = compute_psi_matrix(args, raw, id, condition, method='hilbert')


def analyze_psi_matrix(args, psi_matrix, id, condition):
    """
    
    """
    print(np.min(psi_matrix), np.max(psi_matrix))
    adj1 = apply_thresholding(psi_matrix, threshold_type="thresholded", network_type="weighted", threshold=0.8)

    adj1_results = analyze_network_from_psi(adj1, mode="weighted")
    save_results_to_txt(adj1_results, id, condition, args.connectivity_path, "adj1_result_analyze_network")

    # Load metadata for subject
    meta_info = get_subject_metadata(id, args.metadata_path)
    sex = meta_info["sex"]
    group = meta_info["group"]

    save_global_metrics_csv({
        "global_efficiency": adj1_results['global_efficiency'],
        "local_efficiency": adj1_results['local_efficiency'],
        "density": adj1_results['density']
    }, subject_id=id, sex=sex, group=group, condition=condition, output_path=args.connectivity_path, file_name='adj1_global_network_metrics.csv')

def run_analyze_psi(args):

    subject_ids = get_subject_ids(args.preprocess_path)    

    for id in tqdm(subject_ids, desc="Connectivity Subjects"):
        for condition in ['EO', 'EC']:
            print(f"==={id}_{condition}===")
            
            psi_matrix = load_psi_matrix(args, id, condition)
            print(psi_matrix)
            
            psi_matrix = analyze_psi_matrix(args, psi_matrix, id, condition)





# --- utility functions ---------------------------------------------------

def center_freq(band):
    a, b = band
    return 0.5 * (a + b)

def window_length_from_cycles(freq_hz, cycles=6, min_sec=0.5, max_sec=10.0):
    """Return window length in seconds given center frequency and cycles."""
    if freq_hz <= 0:
        return max(min_sec, 1.0)
    wl = cycles / freq_hz
    wl = max(wl, min_sec)
    wl = min(wl, max_sec)
    return wl

def fir_filter_data(data, sfreq, l_freq, h_freq):
    """
    data: ndarray (n_channels, n_times)
    returns filtered data (same shape) using mne.filter.filter_data (FIR zero-phase)
    """
    # mne.filter.filter_data expects shape (n_channels, n_times)
    filtered = mne.filter.filter_data(
        data.copy(), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
        method='fir', phase='zero', verbose=False
    )
    return filtered

# --- synchronization measures ---------------------------------------------

def compute_phase(data_win):
    """analytic signal phase for data_win shape (n_channels, n_times_window)."""
    return np.angle(hilbert(data_win, axis=1))

def psi_between_phase(phase_i, phase_j):
    """phase_i, phase_j: vectors (n_times,)
       return PSI (1:1)"""
    phase_diff = phase_j - phase_i
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def pli_between_phase(phase_i, phase_j):
    """PLI = |mean(sign(sin(delta_phase)))| (equivalently sign of imaginary part)"""
    phase_diff = phase_j - phase_i
    return np.abs(np.mean(np.sign(np.sin(phase_diff))))

def wpli_between_analytic(x, y):
    """wPLI on analytic signals x,y (shape n_times). Uses sample-wise imag(X * conj(Y))."""
    im = np.imag(x * np.conj(y))
    denom = np.mean(np.abs(im))
    if denom == 0:
        return 0.0
    num = np.abs(np.mean(np.sign(im) * np.abs(im)))
    return num / denom

def imag_coherence_between_analytic(x, y):
    """Imaginary coherence (normalized imaginary part of cross-spectrum)."""
    # sample-wise cross-spectral term
    cross = x * np.conj(y)
    im_mean = np.abs(np.mean(np.imag(cross)))
    pxx = np.mean(np.abs(x) ** 2)
    pyy = np.mean(np.abs(y) ** 2)
    denom = np.sqrt(pxx * pyy)
    if denom == 0:
        return 0.0
    return im_mean / denom

# --- main pipeline -------------------------------------------------------

def compute_dynamic_measures(
    args,
    id,
    condition,
    data,               # (n_channels, n_times)
    sfreq,
    ch_names,
    band_list = ["delta", "theta", "alpha", "beta"],
    cycles = 6,
    overlap = 0.5,
    m_max_extra = 3,    # extra margin when estimating m_max
    min_win_sec = 0.5,
    max_win_sec = 10.0,
    verbose = True
):
    """
    Compute dynamic PSI/PLI/wPLI/imag_coh for same-band and cross-band (n:m).
    Returns a dict with keys:
      - 'per_pair': dict[(b1,b2)] -> dict with:
            'windows' : list of window center times (sec)
            'psi_windows': ndarray (n_windows, n_ch, n_ch)
            'psi_mean'   : ndarray (n_ch, n_ch)   # average across windows
            similarly for 'pli', 'wpli', 'imcoh'
            if b1 != b2: includes 'best_m_windows' list and 'best_m_mean'
    """

    n_ch, n_times = data.shape
    results = {"per_pair": {}}

    # Precompute center freqs and window lengths (use lower-band when cross?)
    band_bounds = {b: BAND_DEFS[b] for b in band_list}
    band_fc = {b: center_freq(band_bounds[b]) for b in band_list}
    band_win = {b: window_length_from_cycles(band_fc[b], cycles, min_win_sec, max_win_sec)
                for b in band_list}

    if verbose:
        print("Bands:", band_list)
        for b in band_list:
            print(f"  {b}: {band_bounds[b]}, center {band_fc[b]:.2f} Hz, win {band_win[b]:.2f}s")

    # For each band, pre-filter the entire recording (FIR zero-phase)
    filtered_data = {}
    for b in band_list:
        l, h = band_bounds[b]
        filtered_data[b] = fir_filter_data(data, sfreq, l, h)

    # Determine whether to compute only same-band pairs or also cross-band pairs
    same_band_only = getattr(args, 'same_band_only', True)

    # Build list of pairs to process according to the flag
    pairs = []
    if same_band_only:
        # only same-band (b,b) for each band in band_list
        for b in band_list:
            pairs.append((b, b))
    else:
        # include same-band and cross-band pairs (upper-triangle ordering)
        for idx1, b1 in enumerate(band_list):
            for b2 in band_list[idx1:]:
                pairs.append((b1, b2))

    # For each selected pair:
    for pair_key in pairs:
        b1, b2 = pair_key

        if verbose:
            print(f"\nProcessing pair {pair_key}")

        data1 = filtered_data[b1]  # (n_ch, n_times)
        data2 = filtered_data[b2]

        # choose window length: if same-band, use that band's win;
        # if cross-band, use the longer of the two (to capture slow cycles)
        if b1 == b2:
            win_sec = band_win[b1]
        else:
            # use max of both windows to ensure adequate cycles of the lower band
            win_sec = max(band_win[b1], band_win[b2])

        step_sec = win_sec * (1.0 - overlap)
        n_win = int(np.floor((n_times / sfreq - win_sec) / step_sec) + 1)
        if n_win < 1:
            # fallback: single window
            n_win = 1
            step_sec = 0
        if verbose:
            print(f"  window {win_sec:.3f}s step {step_sec:.3f}s -> n_win {n_win}")

        # prepare containers
        psi_windows = np.zeros((n_win, n_ch, n_ch))
        pli_windows = np.zeros((n_win, n_ch, n_ch))
        wpli_windows = np.zeros((n_win, n_ch, n_ch))
        imcoh_windows = np.zeros((n_win, n_ch, n_ch))
        best_m_windows = []  # per-window best m for cross-band; for same-band keep 1

        # precompute time indices for windows
        win_samples = int(round(win_sec * sfreq))
        step_samples = int(round(step_sec * sfreq)) if step_sec > 0 else 0
        win_centers = []

        for w in trange(n_win, desc=f"{b1}-{b2}", disable=not verbose):
            start_samp = w * step_samples
            end_samp = start_samp + win_samples
            if end_samp > n_times:
                # pad or clip: clip here
                start_samp = max(0, n_times - win_samples)
                end_samp = n_times
            win_centers.append((start_samp + end_samp) / 2.0 / sfreq)

            x1 = data1[:, start_samp:end_samp]  # (n_ch, win_samples)
            x2 = data2[:, start_samp:end_samp]

            # analytic signals
            a1 = hilbert(x1, axis=1)   # complex
            a2 = hilbert(x2, axis=1)

            # phases (n_ch, win_samples)
            ph1 = np.angle(a1)
            ph2 = np.angle(a2)

            # If same band, simple 1:1
            if b1 == b2:
                for i in range(n_ch):
                    for j in range(n_ch):
                        psi_windows[w, i, j] = psi_between_phase(ph1[i], ph2[j])
                        pli_windows[w, i, j] = pli_between_phase(ph1[i], ph2[j])
                        wpli_windows[w, i, j] = wpli_between_analytic(a1[i], a2[j])
                        imcoh_windows[w, i, j] = imag_coherence_between_analytic(a1[i], a2[j])
                best_m_windows.append(1)
            else:
                # cross-band: sweep m for n=1
                f1c = band_fc[b1]
                f2c = band_fc[b2]
                # rough m_max: ceil(f2c/f1c) + margin
                if f1c == 0:
                    m_max = 1
                else:
                    m_max = int(np.ceil(f2c / f1c)) + m_max_extra
                    if m_max < 1:
                        m_max = 1
                best_m_for_window = np.ones((n_ch, n_ch), dtype=int) * 1
                best_psi_for_window = np.zeros((n_ch, n_ch))

                # For efficiency: compute phase arrays once per channel pair inside loops
                for i in range(n_ch):
                    for j in range(n_ch):
                        # sweep m
                        best_psi = -1.0
                        best_m = 1
                        for m in range(1, m_max + 1):
                            phase_diff = 1 * ph2[j] - m * ph1[i]
                            psi_val = np.abs(np.mean(np.exp(1j * phase_diff)))
                            if psi_val > best_psi:
                                best_psi = psi_val
                                best_m = m
                        psi_windows[w, i, j] = best_psi
                        best_m_for_window[i, j] = best_m
                        # compute other metrics using analytic signals at 1:1 alignment (not n:m):
                        # For connectivity metrics like wPLI/PLI/imcoh we normally compute narrowband between
                        # the same filtered signals, so we compute them here (they reflect band-band interactions)
                        pli_windows[w, i, j] = pli_between_phase(ph1[i], ph2[j])
                        wpli_windows[w, i, j] = wpli_between_analytic(a1[i], a2[j])
                        imcoh_windows[w, i, j] = imag_coherence_between_analytic(a1[i], a2[j])

                # store best_m matrix averaged (or keep full matrix)
                # here we store the per-window average best m (rounded) for convenience
                best_m_windows.append(best_m_for_window)

        # average across windows
        psi_mean = np.mean(psi_windows, axis=0)
        pli_mean = np.mean(pli_windows, axis=0)
        wpli_mean = np.mean(wpli_windows, axis=0)
        imcoh_mean = np.mean(imcoh_windows, axis=0)

        results["per_pair"][pair_key] = {
            "windows": win_centers,
            "psi_windows": psi_windows,
            "psi_mean": psi_mean,
            "pli_windows": pli_windows,
            "pli_mean": pli_mean,
            "wpli_windows": wpli_windows,
            "wpli_mean": wpli_mean,
            "imcoh_windows": imcoh_windows,
            "imcoh_mean": imcoh_mean,
            "best_m_windows": best_m_windows,
        }

        # if pair was not symmetrical (b1 != b2), also store reversed key for convenience
        if b1 != b2:
            results["per_pair"][(b2, b1)] = results["per_pair"][pair_key]

        filename = f"psi_{pair_key[0]}_{pair_key[1]}_{id}_{condition}.pkl"
        save_dir_path = os.path.join(args.pair_result_path, id)

        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(save_dir_path, filename)
        with open(save_path, "wb") as f:
            pickle.dump(results["per_pair"][pair_key], f)

        plot_save_psi_matrix_2(results["per_pair"][pair_key], pair_key, ch_names, id, condition, args.psi_path)

    return results

def plot_save_psi_matrix_2(psi_matrix, pair_key, ch_names, id, condition, output_path):
    for metric in ["psi_mean", "pli_mean", "wpli_mean", "imcoh_mean"]:
        plt.figure(figsize=(10, 8))
        sns.heatmap(psi_matrix[metric], xticklabels=ch_names, yticklabels=ch_names,
                cmap='viridis', center=0, annot=False, fmt=".2f", square=True)
        plt.title(f'{metric} ({id}, {condition})')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.tight_layout()

        # Save the figure
        subject_dir = os.path.join(output_path, id)
        os.makedirs(subject_dir, exist_ok=True)
        img_filename = os.path.join(subject_dir, f"{metric}_{pair_key[0]}_{pair_key[1]}_{id}_{condition}.png")
        plt.savefig(img_filename, dpi=300)
        plt.close()
        print(f"Saved {metric} plot to {img_filename}")

def _phase_randomize_channel(sig, random_state=None):
    """Phase-randomize a single real signal while preserving the amplitude spectrum.
    Returns a real-valued surrogate time series of same length.
    """
    n = sig.shape[0]
    X = np.fft.fft(sig)

    rng = np.random.default_rng(random_state)

    # indices for positive frequencies (exclude DC=0 and Nyquist if present)
    if n % 2 == 0:
        pos_idx = np.arange(1, n // 2)
    else:
        pos_idx = np.arange(1, (n + 1) // 2)

    phases = rng.uniform(0, 2 * np.pi, size=pos_idx.shape[0])

    # apply random phases to positive freqs and enforce Hermitian symmetry
    X_sur = X.copy()
    X_sur[pos_idx] = X[pos_idx] * np.exp(1j * phases)
    X_sur[-pos_idx] = np.conj(X_sur[pos_idx])

    # leave DC and Nyquist unchanged
    x_surr = np.fft.ifft(X_sur).real
    return x_surr


def phase_randomize_signals(data, random_state=None):
    """Phase-randomize multichannel data (n_ch, n_times). Returns surrogate array.
    """
    n_ch, n_times = data.shape
    surr = np.zeros_like(data)
    for ch in range(n_ch):
        surr[ch] = _phase_randomize_channel(data[ch], random_state=random_state)
    return surr


# --- Progress plotting utility -------------------------------------------------
class ProgressPlotter:
    """Simple progress visualizer that saves a PNG with horizontal bars for each pair.

    Usage:
      pp = ProgressPlotter(out_dir, pairs)
      pp.set_total(n_surrogates)
      pp.update_surrogate(pair_key_str, current_sur, total_sur)
      pp.mark_pair_done(pair_key_str)
    """
    def __init__(self, out_dir, pairs):
        self.out_dir = out_dir
        self.pairs = [f"{p[0]}-{p[1]}" for p in pairs]
        self.progress = {k: 0.0 for k in self.pairs}
        self.total = None
        os.makedirs(self.out_dir, exist_ok=True)
        self.png_path = os.path.join(self.out_dir, 'surrogate_progress.png')

    def set_total(self, n_surrogates):
        self.total = int(n_surrogates)
        self._draw()

    def update_surrogate(self, pair_key_str, current, total):
        try:
            frac = float(current) / float(total)
        except Exception:
            frac = 0.0
        if pair_key_str in self.progress:
            self.progress[pair_key_str] = min(1.0, max(0.0, frac))
        self._draw()

    def mark_pair_done(self, pair_key_str):
        if pair_key_str in self.progress:
            self.progress[pair_key_str] = 1.0
        self._draw()

    def _draw(self):
        try:
            import matplotlib.pyplot as plt
            labels = list(self.progress.keys())
            vals = [self.progress[k] for k in labels]
            y = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(6, max(2, 0.3 * len(labels))))
            ax.barh(y, vals, color='C0')
            ax.set_xlim(0, 1)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Progress')
            for i, v in enumerate(vals):
                ax.text(v + 0.02, i, f"{int(v*100)}%", va='center', fontsize=8)
            plt.tight_layout()
            fig.savefig(self.png_path, dpi=150)
            plt.close(fig)
        except Exception:
            # silently ignore plotting errors (progress is auxiliary)
            pass


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction for a 2D p-value matrix.
    Returns boolean mask of significant entries (same shape as pvals).
    """
    p = pvals.flatten()
    n = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, n+1) / n) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros_like(pvals, dtype=bool)
    max_idx = np.max(np.where(below)[0])
    cutoff = sorted_p[max_idx]
    sig_mask = pvals <= cutoff
    return sig_mask


def surrogate_psi_test_for_pair(
    data,
    sfreq,
    pair_key,
    band_list,
    n_surrogates=200,
    cycles=6,
    overlap=0.5,
    m_max_extra=3,
    min_win_sec=0.5,
    max_win_sec=10.0,
    random_state=None,
    alpha=0.05,
    verbose=True,
    show_progress=False,
    progress_position=0,
    progress_plotter=None,
    progress_key=None,
    n_jobs=1,
    chunk_size=10,
    save_surrogates=False,
):
    """Perform phase-randomization surrogate test for PSI mean matrix of a band pair.

    This implementation parallelizes surrogate generation in chunks and accumulates
    counts of how often each surrogate >= observed to compute p-values without
    keeping the full surrogate set in memory. Use `n_jobs` (joblib loky backend)
    and `chunk_size` to tune parallel efficiency. If `save_surrogates` is True,
    a (potentially large) array of surrogates will be returned; otherwise None.
    """
    import time
    rng = np.random.default_rng(random_state)

    b1, b2 = pair_key
    if b1 not in BAND_DEFS or b2 not in BAND_DEFS:
        raise ValueError("pair_key must be band names defined in BAND_DEFS")

    band_bounds_local = {b: BAND_DEFS[b] for b in band_list}
    band_fc = {b: center_freq(band_bounds_local[b]) for b in band_list}
    band_win = {b: window_length_from_cycles(band_fc[b], cycles, min_win_sec, max_win_sec)
                for b in band_list}

    # choose window length
    if b1 == b2:
        win_sec = band_win[b1]
    else:
        win_sec = max(band_win[b1], band_win[b2])

    step_sec = win_sec * (1.0 - overlap)
    n_ch, n_times = data.shape
    step_samples = int(round(step_sec * sfreq)) if step_sec > 0 else 0
    win_samples = int(round(win_sec * sfreq))

    n_win = int(np.floor((n_times / sfreq - win_sec) / step_sec) + 1) if step_sec > 0 else 1
    if n_win < 1:
        n_win = 1

    if verbose:
        print(f"Surrogate test for pair {pair_key}: win {win_sec:.3f}s step {step_sec:.3f}s n_win {n_win} n_surrogates {n_surrogates} n_jobs {n_jobs} chunk_size {chunk_size}")

    # filter original data per band (use same fir_filter_data)
    l1, h1 = BAND_DEFS[b1]
    l2, h2 = BAND_DEFS[b2]
    filt1 = fir_filter_data(data, sfreq, l1, h1)
    filt2 = fir_filter_data(data, sfreq, l2, h2)

    # compute observed psi_mean following same algorithm as compute_dynamic_measures for this pair
    psi_windows = np.zeros((n_win, n_ch, n_ch))
    for w in range(n_win):
        start = int(w * step_samples)
        end = start + win_samples
        if end > n_times:
            start = max(0, n_times - win_samples)
            end = n_times
        x1 = filt1[:, start:end]
        x2 = filt2[:, start:end]
        a1 = hilbert(x1, axis=1)
        a2 = hilbert(x2, axis=1)
        ph1 = np.angle(a1)
        ph2 = np.angle(a2)

        if b1 == b2:
            for i in range(n_ch):
                for j in range(n_ch):
                    psi_windows[w, i, j] = psi_between_phase(ph1[i], ph2[j])
        else:
            f1c = band_fc[b1]
            f2c = band_fc[b2]
            if f1c == 0:
                m_max = 1
            else:
                m_max = int(np.ceil(f2c / f1c)) + m_max_extra
                if m_max < 1:
                    m_max = 1
            for i in range(n_ch):
                for j in range(n_ch):
                    best_psi = -1.0
                    for m in range(1, m_max + 1):
                        phase_diff = 1 * ph2[j] - m * ph1[i]
                        psi_val = np.abs(np.mean(np.exp(1j * phase_diff)))
                        if psi_val > best_psi:
                            best_psi = psi_val
                    psi_windows[w, i, j] = best_psi

    observed_mean = np.mean(psi_windows, axis=0)

    # Prepare seeds for reproducible per-surrogate RNG
    seeds = [int(rng.integers(1_000_000_000)) for _ in range(n_surrogates)]

    # accumulator for counts of surrogate >= observed (one-sided test)
    counts = np.zeros((n_ch, n_ch), dtype=np.int64)

    # optional container for saved surrogates
    surrogates_saved = [] if save_surrogates else None

    # helper: compute one surrogate psi_mean given seed
    def _compute_one_surrogate(seed):
        # phase-randomize
        data_surr = phase_randomize_signals(data, random_state=seed)
        # filter
        filt1_s = fir_filter_data(data_surr, sfreq, l1, h1)
        filt2_s = fir_filter_data(data_surr, sfreq, l2, h2)
        # compute windows mean
        psi_w_s = np.zeros((n_win, n_ch, n_ch))
        for w in range(n_win):
            start = int(w * step_samples)
            end = start + win_samples
            if end > n_times:
                start = max(0, n_times - win_samples)
                end = n_times
            x1 = filt1_s[:, start:end]
            x2 = filt2_s[:, start:end]
            a1 = hilbert(x1, axis=1)
            a2 = hilbert(x2, axis=1)
            ph1 = np.angle(a1)
            ph2 = np.angle(a2)

            if b1 == b2:
                for i in range(n_ch):
                    for j in range(n_ch):
                        psi_w_s[w, i, j] = psi_between_phase(ph1[i], ph2[j])
            else:
                f1c = band_fc[b1]
                f2c = band_fc[b2]
                if f1c == 0:
                    m_max = 1
                else:
                    m_max = int(np.ceil(f2c / f1c)) + m_max_extra
                    if m_max < 1:
                        m_max = 1
                for i in range(n_ch):
                    for j in range(n_ch):
                        best_psi = -1.0
                        for m in range(1, m_max + 1):
                            phase_diff = 1 * ph2[j] - m * ph1[i]
                            psi_val = np.abs(np.mean(np.exp(1j * phase_diff)))
                            if psi_val > best_psi:
                                best_psi = psi_val
                        psi_w_s[w, i, j] = best_psi
        return np.mean(psi_w_s, axis=0)

    # run surrogates in chunks to avoid keeping all surrogates in memory
    import time
    t0 = time.time()
    for start in range(0, n_surrogates, chunk_size):
        end = min(n_surrogates, start + chunk_size)
        seed_chunk = seeds[start:end]
        # parallel compute per-chunk
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_compute_one_surrogate)(s) for s in seed_chunk
        )
        # update counts and optionally save
        for psi_s in results:
            counts += (psi_s >= observed_mean).astype(np.int64)
            if save_surrogates:
                surrogates_saved.append(psi_s)
        if verbose:
            done = end
            print(f"  processed {done}/{n_surrogates} surrogates (elapsed {time.time()-t0:.1f}s)")
        # update progress plot if available
        if progress_plotter is not None and progress_key is not None:
            try:
                progress_plotter.update_surrogate(progress_key, min(n_surrogates, end), n_surrogates)
            except Exception:
                pass

    # build p-values (one-sided)
    p_matrix = (counts + 1.0) / (n_surrogates + 1.0)

    # FDR correction
    sig_mask = benjamini_hochberg(p_matrix, alpha=alpha)

    # convert saved surrogates list to array if requested
    surrogates_array = None
    if save_surrogates and len(surrogates_saved) > 0:
        surrogates_array = np.stack(surrogates_saved, axis=0)

    return observed_mean, p_matrix, sig_mask, surrogates_array
def run_surrogate_and_plot_for_subject(args, subject_id, condition, data, sfreq, results, ch_names, n_surrogates=500):
    """For a given subject and condition, run surrogate tests for selected band pairs
    (controlled by args.same_band_only) and produce connectivity circle plots for
    significant connections only. Processing is parallelized across band pairs using joblib.

    Saves observed, p-values, sig mask, optional surrogates, and PNG circle plots
    under args.pair_result_path/<subject_id>/.
    """
    out_dir = os.path.join(args.pair_result_path, subject_id)
    os.makedirs(out_dir, exist_ok=True)

    # determine pairs to run according to args.same_band_only
    band_list = getattr(args, 'band_list', ["delta","theta","alpha","beta"]) if hasattr(args, 'band_list') else ["delta","theta","alpha","beta"]
    same_band_only = getattr(args, 'same_band_only', True)

    pairs = []
    if same_band_only:
        pairs = [(b, b) for b in band_list]
    else:
        for i, b1 in enumerate(band_list):
            for b2 in band_list[i:]:
                pairs.append((b1, b2))

    # joblib parallel settings
    n_jobs = getattr(args, 'n_jobs', 4)  # -1 = use all cores by default
    backend = getattr(args, 'parallel_backend', 'loky')

    # optional visual progress
    show_progress_plot = getattr(args, 'show_progress_plot', False)
    progress_plotter = ProgressPlotter(out_dir, pairs) if show_progress_plot else None

    def _process_pair(pair_key, show_progress=False, progress_position=0, progress_plotter=None, progress_key=None):
        try:
            if getattr(args, 'verbose', False):
                print(f"Surrogate+plot for {subject_id} {condition} pair {pair_key}")

            observed_mean, p_matrix, sig_mask, surrogates = surrogate_psi_test_for_pair(
                data, sfreq, pair_key, band_list,
                n_surrogates=n_surrogates,
                cycles=getattr(args, 'cycles', 6),
                overlap=getattr(args, 'overlap', 0.5),
                m_max_extra=getattr(args, 'm_max_extra', 3),
                min_win_sec=getattr(args, 'min_win_sec', 0.5),
                max_win_sec=getattr(args, 'max_win_sec', 10.0),
                random_state=getattr(args, 'random_state', None),
                alpha=getattr(args, 'alpha', 0.05),
                verbose=getattr(args, 'verbose', False),
                show_progress=show_progress,
                progress_position=progress_position,
                progress_plotter=progress_plotter,
                progress_key=progress_key
            )

            # Only keep significant connections
            adj_sig = np.where(sig_mask, observed_mean, 0.0)

            # Save matrices
            base = f"surrogate_{pair_key[0]}_{pair_key[1]}_{subject_id}_{condition}"
            np.save(os.path.join(out_dir, f"{base}_observed.npy"), observed_mean)
            np.save(os.path.join(out_dir, f"{base}_p.npy"), p_matrix)
            np.save(os.path.join(out_dir, f"{base}_sigmask.npy"), sig_mask)
            if getattr(args, 'save_surrogates', False):
                np.save(os.path.join(out_dir, f"{base}_samples.npy"), surrogates)

            # Create connectivity circle plot for significant edges only
            try:
                vmax = np.max(np.abs(observed_mean))
                if vmax == 0:
                    vmax = 1.0
                # use plot_connectivity_circle from mne_connectivity.viz (imported as plot_connectivity_circle)
                fig = plot_connectivity_circle(
                    adj_sig, ch_names, title=f"{subject_id} {condition} {pair_key} (sig only)",
                    colormap='hot', vmin=0.0, vmax=vmax, show=False
                )
                img_path = os.path.join(out_dir, f"connectivity_circle_{pair_key[0]}_{pair_key[1]}_{subject_id}_{condition}.png")
                try:
                    fig.savefig(img_path, dpi=300)
                    plt.close(fig)
                except Exception:
                    plt.savefig(img_path, dpi=300)
                    plt.close()
                if getattr(args, 'verbose', False):
                    print(f"Saved connectivity circle to {img_path}")
            except Exception as e:
                print(f"Failed to plot connectivity circle for {pair_key}: {e}")

            # return results to attach in main results dict
            return (pair_key, observed_mean, p_matrix, sig_mask)

        except Exception as e:
            print(f"Surrogate test/plot failed for {subject_id} {condition} pair {pair_key}: {e}")
            return (pair_key, None, None, None)

    # If running single-job (n_jobs == 1), run serially so we can show per-surrogate tqdm bars
    processed = []
    if n_jobs == 1:
        if getattr(args, 'verbose', False):
            print("Running surrogate tests serially (n_jobs=1) — per-surrogate progress bars enabled.")
        for idx, pk in enumerate(pairs):
            # use a fixed progress_position so bars are placed below outer loop
            progress_key = f"{pk[0]}-{pk[1]}" if progress_plotter is not None else None
            res = _process_pair(pk, show_progress=True, progress_position=1, progress_plotter=progress_plotter, progress_key=progress_key)
            processed.append(res)
    else:
        if getattr(args, 'verbose', False):
            print("Running surrogate tests in parallel — per-surrogate tqdm disabled. Use n_jobs=1 to enable detailed progress.")
        # Run tasks in a ThreadPoolExecutor so we can show a simple per-pair progress bar
        if getattr(args, 'verbose', False):
            print("Running surrogate tests in parallel — showing per-pair progress (threads used).")

        # determine number of workers
        if n_jobs == -1:
            max_workers = os.cpu_count() or 1
        else:
            max_workers = max(1, int(n_jobs))

        processed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # submit all pair tasks
            future_to_pair = {executor.submit(_process_pair, pk): pk for pk in pairs}
            # iterate as they complete and update a simple tqdm counter
            for fut in tqdm(concurrent.futures.as_completed(future_to_pair),
                            total=len(future_to_pair), desc="pairs", unit="pair", disable=not getattr(args, 'verbose', False)):
                try:
                    res = fut.result()
                except Exception as e:
                    pk = future_to_pair.get(fut)
                    print(f"Error processing pair {pk}: {e}")
                    res = (pk, None, None, None)
                processed.append(res)
                # update progress plot per-pair (parallel case)
                if progress_plotter is not None:
                    try:
                        pair_key = future_to_pair.get(fut)
                        if pair_key is not None:
                            progress_plotter.mark_pair_done(f"{pair_key[0]}-{pair_key[1]}")
                    except Exception:
                        pass

    # attach to results and save per-pair files
    for pair_key, observed_mean, p_matrix, sig_mask in processed:
        if observed_mean is None:
            continue
        # ensure container exists
        if pair_key not in results.get('per_pair', {}):
            results.setdefault('per_pair', {})[pair_key] = {}
        results['per_pair'][pair_key]['surrogate_observed'] = observed_mean
        results['per_pair'][pair_key]['surrogate_p'] = p_matrix
        results['per_pair'][pair_key]['surrogate_sigmask'] = sig_mask

    # save per-pair summary
    summary_path = os.path.join(out_dir, f"per_pair_results_{subject_id}_{condition}.pkl")
    with open(summary_path, 'wb') as sf:
        pickle.dump(results.get('per_pair', {}), sf)
    print(f"Saved per-pair summary to {summary_path}")

def run_psi_2(args):
    """Compute dynamic PSI/PLI/wPLI/imcoh for all subjects and optionally run surrogates.

    Behavior:
      - Iterates subjects from args.preprocess_path (via get_subject_ids)
      - For each subject/condition, loads preprocessed raw .fif, picks EEG channels
      - Applies compute_csd (same as compute_psi_matrix did)
      - Calls compute_dynamic_measures(...) to compute per-pair windowed and mean metrics
      - Saves the full results dict to args.pair_result_path/<subject_id>/dynamic_results_<id>_<condition>.pkl
      - If args.n_surrogates > 0, calls run_surrogate_and_plot_for_subject(...) to run surrogate testing

    Notes:
      - This function expects args to provide at least: preprocess_path, pair_result_path
      - Surrogate-related options are read from args and forwarded to the surrogate runner.
    """
    subject_ids = get_subject_ids(args.preprocess_path)

    for subject_id in tqdm(subject_ids, desc="Connectivity Subjects (dynamic)"):
        for condition in ['EO', 'EC']:
            print(f"==={subject_id}_{condition}===")
            # Load raw file
            path = os.path.join(args.preprocess_path, subject_id, f"{subject_id}_{condition}_eeg.fif")
            if not os.path.exists(path):
                print(f"Raw EEG file for condition {condition} does not exist: {path}")
                continue

            try:
                raw = mne.io.read_raw_fif(path, preload=True)
            except Exception as e:
                print(f"Failed to read raw file {path}: {e}")
                continue

            # pick EEG channels
            try:
                raw.pick('eeg')
            except Exception:
                # if pick fails, continue with raw as-is
                pass

            # apply compute_csd if available (keeps behavior consistent with compute_psi_matrix)
            try:
                raw = compute_csd(raw)
            except Exception as e:
                if getattr(args, 'verbose', False):
                    print(f"compute_csd failed or skipped for {subject_id} {condition}: {e}")

            # prepare data array for dynamic measures
            try:
                data = raw.get_data()  # shape (n_channels, n_times)
                sfreq = raw.info.get('sfreq', None)
                ch_names = raw.ch_names
            except Exception as e:
                print(f"Failed to extract data from raw for {subject_id} {condition}: {e}")
                continue

            # compute dynamic measures
            try:
                results = compute_dynamic_measures(
                    args=args,
                    id=subject_id,
                    condition=condition,
                    data=data,
                    sfreq=sfreq,
                    ch_names=ch_names,
                    band_list=getattr(args, 'band_list', ["delta", "theta", "alpha", "beta"]),
                    cycles=getattr(args, 'cycles', 6),
                    overlap=getattr(args, 'overlap', 0.5),
                    m_max_extra=getattr(args, 'm_max_extra', 3),
                    min_win_sec=getattr(args, 'min_win_sec', 0.5),
                    max_win_sec=getattr(args, 'max_win_sec', 10.0),
                    verbose=getattr(args, 'verbose', False)
                )
            except Exception as e:
                print(f"Failed compute_dynamic_measures for {subject_id} {condition}: {e}")
                continue

            # save dynamic results per subject/condition
            out_dir = os.path.join(args.pair_result_path, subject_id)
            os.makedirs(out_dir, exist_ok=True)
            dyn_path = os.path.join(out_dir, f"dynamic_results_{subject_id}_{condition}.pkl")
            try:
                with open(dyn_path, 'wb') as f:
                    pickle.dump(results, f)
                if getattr(args, 'verbose', False):
                    print(f"Saved dynamic results to {dyn_path}")
            except Exception as e:
                print(f"Failed to save dynamic results for {subject_id} {condition}: {e}")

            # Optionally run surrogates if requested
            n_surrogates = getattr(args, 'n_surrogates', 0)
            if n_surrogates and int(n_surrogates) > 0:
                if getattr(args, 'verbose', False):
                    print(f"Running surrogate tests (n_surrogates={n_surrogates}) for {subject_id} {condition}")
                try:
                    run_surrogate_and_plot_for_subject(
                        args=args,
                        subject_id=subject_id,
                        condition=condition,
                        data=data,
                        sfreq=sfreq,
                        results=results,
                        ch_names=ch_names,
                        n_surrogates=int(n_surrogates)
                    )
                except Exception as e:
                    print(f"Surrogate runner failed for {subject_id} {condition}: {e}")

    print("run_psi_2 finished.")
    