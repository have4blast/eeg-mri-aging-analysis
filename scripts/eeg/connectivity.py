import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
from tqdm import tqdm
from tqdm import trange
import pickle

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

    # For each pair:
    for idx1, b1 in enumerate(band_list):
        for idx2, b2 in enumerate(band_list[idx1:]):
            # ensure ordered pair (b1,b2) where idx2 >= idx1
            # we'll store both (b1,b2) and (b2,b1) if needed later
            if idx1 == idx2:
                pair_key = (b1, b1)
            else:
                pair_key = (b1, b2)

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

            plot_save_psi_matrix_2(results["per_pair"][pair_key]["psi_mean"], pair_key, data.ch_names, id, condition, args.psi_path)
            plot_save_psi_matrix_2(results["per_pair"][pair_key]["pli_mean"], pair_key, data.ch_names, id, condition, args.psi_path)
            plot_save_psi_matrix_2(results["per_pair"][pair_key]["wpli_mean"], pair_key, data.ch_names, id, condition, args.psi_path)
            plot_save_psi_matrix_2(results["per_pair"][pair_key]["imcoh_mean"], pair_key, data.ch_names, id, condition, args.psi_path)

    return results

def plot_save_psi_matrix_2(psi_matrix, pair_key, ch_names, id, condition, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(psi_matrix, xticklabels=ch_names, yticklabels=ch_names,
            cmap='viridis', center=0, annot=False, fmt=".2f", square=True)
    plt.title(f'Phase Synchronization Index ({id}, {condition})')
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    plt.tight_layout()
    
    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(subject_dir), exist_ok=True)
    
    # Save the figure
    img_filename = os.path.join(subject_dir, f"psi_{pair_key[0]}_{pair_key[1]}_{id}_{condition}.png")

    plt.savefig(img_filename, dpi=300)
    plt.close()
    print(f"Saved PSI matrix plot to {img_filename}")

def run_psi_2(args):

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
            # if not raw.preload:
            #     raw.load_data()

            data = raw.get_data()  # shape: (n_channels, n_times)

            sfreq = raw.info["sfreq"]

            band_list = ["delta", "theta", "alpha", "beta"]
            results = compute_dynamic_measures(
            args,
            id,
            condition,
            data=data,
            sfreq=sfreq,
            band_list=band_list,
            cycles=6,
            overlap=0.5,
            verbose=True
            )


# def estimate_m_range(phase1_band, phase2_band):
#     """
#     Estimate the range of m such that m * phase1_freq approximates phase2_freq.
#     Inputs are tuples representing frequency bands, e.g., (0.5, 4) for delta.
#     Returns integer m_min and m_max.
#     """
#     f1_min, f1_max = phase1_band
#     f2_min, f2_max = phase2_band

#     # Avoid division by zero
#     if f1_max == 0 or f1_min == 0:
#         raise ValueError("Frequency band includes 0 Hz, which is invalid.")

#     m_min = round(np.floor(f2_min / f1_max))  # e.g. phase1 maxが4Hz, phase2 minが4Hzなら m_min = 1
#     m_max = round(np.ceil(f2_max / f1_min))   # e.g. phase1 minが0.5Hz, phase2 maxが8Hzなら m_max = 16

#     # Ensure m_min is at least 1
#     m_min = max(1, m_min)

#     return m_min, m_max

# def find_best_m_for_n1(phase1, phase2, m_range=(1, 10), plot=False):
#     """
#     Find the best m value for fixed n=1 that maximizes phase-locking value (PLV)
    
#     Parameters:
#         phase1 (ndarray): Phase time series of the low-frequency signal
#         phase2 (ndarray): Phase time series of the high-frequency signal
#         m_range (tuple): Range of m values to test (min, max), inclusive
    
#     Returns:
#         best_m (int): Value of m with highest PLV for n=1
#         psi_values (ndarray): PSI values for each m in the given range
#     """
#     n = 1
#     #m_vals = np.arange(m_range[0], m_range[1] + 0.1, 0.1)
#     m_vals = np.arange(m_range[0], m_range[1] + 1, 1)
#     psi_values = []

#     for m in tqdm(m_vals, desc="Searching best m", leave=False):
#         # Compute phase difference for n=1:m
#         phase_diff = n * phase2 - m * phase1
#         # Compute PSI for the current m
#         psi = np.abs(np.mean(np.exp(1j * phase_diff)))
#         psi_values.append(psi)

#     psi_values = np.array(psi_values)
#     best_psi = np.argmax(psi_values)
#     best_m = m_vals[best_psi]

#     # Plot the results
#     if plot:
#         plt.figure(figsize=(8, 5))
#         plt.plot(m_vals, psi_values, marker='o')
#         plt.title('Phase Synchronization Index (PSI) for n=1 and varying m')
#         plt.xlabel('m (harmonic of wave)')
#         plt.ylabel('Phase Synchronization Index (PSI)')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     return best_m, best_psi

# def get_phase(args, raw1, raw2, f1, f2, method='wavelet'):
#     """    Extract phase information from two raw EEG datasets using either Hilbert transform or wavelet transform. 
#     Parameters:
#         raw1 (mne.io.Raw): First raw EEG dataset.
#         raw2 (mne.io.Raw): Second raw EEG dataset.
#         f1 (tuple): Frequency band for the first dataset (e.g., theta).
#         f2 (tuple): Frequency band for the second dataset (e.g., alpha).
#         method (str): Method to use for phase extraction ('hilbert' or 'wavelet').
#     Returns:
#         phase1 (ndarray): Phase of the first dataset, shape: (n_channels, n_times).
#         phase2 (ndarray): Phase of the second dataset, shape: (n_channels, n_times).
#     """
#     if method == 'hilbert':
#         phase1 = np.angle(hilbert(raw1.get_data(), axis=1))
#         phase2 = np.angle(hilbert(raw2.get_data(), axis=1))
#     elif method == 'wavelet':
    
#         epochs1 = create_epochs(raw1, args.epoch_duration)
#         epochs2 = create_epochs(raw2, args.epoch_duration)

#         freqs1 = np.linspace(f1[0], f1[1], 5)
#         freqs2 = np.linspace(f2[0], f2[1], 5)

#         # Phase1 extraction
#         phase1 = tfr_array_morlet(epochs1, sfreq=raw1.info['sfreq'],
#                                   freqs=freqs1, n_cycles=freqs1 / 2, output='phase')[0]
#         phase1 = np.mean(phase1, axis=1)  # shape: (n_channels, n_times)

#         # Phase2 extraction
#         phase2 = tfr_array_morlet(epochs2, sfreq=raw2.info['sfreq'],
#                                   freqs=freqs2, n_cycles=freqs2 / 2, output='phase')[0]
#         phase2 = np.mean(phase2, axis=1)
#     else:
#         raise ValueError("method must be 'hilbert' or 'wavelet'")
    
#     return phase1, phase2

# def plot_save_psi_matrix(psi_matrix, id, condition, best_m, band1, band2, output_path):
#     """
#     Plot the n:m Phase Synchronization Index (PSI) matrix and save it as an image.
#     Parameters:
#         psi_matrix (ndarray): n:m Phase Synchronization Index matrix, shape: (n_channels, n_channels).
#         id (str): Subject ID.
#         condition (str): Condition label (e.g., 'EO', 'EC').
#         best_m (int): Best m value found for the n:m coupling.
#         f1 (str): Name of the first frequency band (e.g., 'delta').
#         f2 (str): Name of the second frequency band (e.g., 'delta').
#         output_path (str): Path to save the output image.
#     """

#     plt.figure(figsize=(6, 5))
#     plt.imshow(psi_matrix, cmap='plasma', interpolation='nearest')
#     plt.colorbar(label='n:m Phase Synchronization Index (PSI)')
#     plt.title(f'1:{best_m} Phase Synchronization Index ({band1} Hz & {band2} Hz)')
#     plt.xlabel('Channel j (Phase2)')
#     plt.ylabel('Channel i (Phase1)')
#     plt.tight_layout()
    
#     subject_dir = os.path.join(output_path, id)

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(subject_dir), exist_ok=True)
    
#     # Save the figure
#     img_filename = os.path.join(subject_dir, f"{id}_{condition}.png")

#     plt.savefig(img_filename, dpi=300)
#     plt.close()
#     print(f"Saved PSI matrix plot to {img_filename}")


# def compute_nm_phase_phase_matrix(args, raw, id, condition, band1, band2, method='wavelef', simulate=False):
#     """
#     Compute n:m Phase Synchronization Index between all pairs of EEG channels.

#     Parameters:
#     -----------
#     args : argparse.Namespace
#         Contains paths and parameters.
#     raw : mne.io.Raw
#         Raw EEG data (preprocessed).
#     id : str
#         Subject ID (e.g., 'sub-010002').
#     condition : str
#         Condition label (e.g., 'EO', 'EC').
#     band1 : str
#         Name of the first frequency band (e.g., 'delta').
#     band2 : str
#         Name of the second frequency band (e.g., 'delta').
#     method : str
#         'hilbert' or 'wavelet'
#     simulate : bool
#         If True, simulate n:m phase coupling for a range of m values.
#     Returns:
#     --------
#     phase_phase_matrix : np.ndarray, shape (n_channels, n_channels)
#         Matrix of n:m Phase Synchronization Index (PSI).
#     """

#     bands = {
#     'delta': (0.5, 4),
#     'theta': (4, 8),
#     'alpha': (8, 13),
#     'beta': (13, 30),
#     'low_gamma': (30, 50),
#     'mid_gamma': (50, 80),
#     'high_gamma': (80, 100)
#     }

#     # Bandpass filtering
#     f1 = bands[band1]
#     f2 = bands[band2]
#     raw1 = raw.copy().filter(f1[0], f1[1], method='iir', verbose=False)
#     raw2 = raw.copy().filter(f2[0], f2[1], method='iir', verbose=False)

#     # Compute CSD for both bands
#     raw1 = compute_csd(raw1)
#     raw2 = compute_csd(raw2)

#     # Get phase information
#     phase1, phase2 = get_phase(args, raw1, raw2, f1, f2, method=method)

#     # Initialize the phase-phase matrix
#     n_channels = phase1.shape[0]
#     psi_matrix = np.zeros((n_channels, n_channels))

#     if simulate:
#         m_min, m_max = estimate_m_range(f1, f2)
#     else:
#         n = 1
#         m = 1

#     # Compute n:m phase-phase coupling for each channel pair
#     for i in tqdm(range(n_channels), desc=f"Computing PSI ({id}, {condition})", leave=True):
#         for j in range(n_channels):
#             if simulate:
#                 # Simulate n:m phase coupling
#                 best_m, best_psi = find_best_m_for_n1(phase1[i], phase2[j], m_range=(m_min, m_max))
#                 psi_matrix[i, j] = best_psi

#             else:
#                 phase_diff = n * phase2 - m * phase1
#                 # Compute PSI for the current m
#                 psi = np.abs(np.mean(np.exp(1j * phase_diff)))
#                 psi_matrix[i, j] = psi

#     psi_matrix = np.array(psi_matrix)
#     psi_matrix = np.mean(psi_matrix, axis=0)

#     # Save the matrix to a file
#     save_psi_matrix(psi_matrix, id, condition, args.psi_path)

#     plot_save_psi_matrix(psi_matrix, id, condition, best_m, band1, band2, args.psi_path)

    
#     return psi_matrix