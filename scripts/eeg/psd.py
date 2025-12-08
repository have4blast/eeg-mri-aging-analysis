import os
import numpy as np
import mne
from tqdm import tqdm
from scripts.eeg.preprocess_run import get_subject_ids
from scripts.eeg.connectivity import get_subject_metadata

def run_psd(args):

    subject_ids = get_subject_ids(args.preprocess_path)    

    for id in tqdm(subject_ids, desc="PSD Subjects"):
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

            raw = raw.copy().pick("eeg")
            meta_info = get_subject_metadata(id, args.metadata_path)
            sex = meta_info["sex"]
            group = meta_info["group"]

            # Compute the Power Spectral Density (PSD) for each channel

            save_global_metrics_csv({
                "global_efficiency": adj3_results['global_efficiency'],
                "local_efficiency": adj3_results['local_efficiency'],
                "density": adj3_results['density']
            }, subject_id=id, sex=sex, group=group, condition=condition, output_path=args.connectivity_path, file_name='global_network_metrics.csv')


def compute_psd_multichannel(raw, fmin=0.5, fmax=45):
    """
    Compute the Power Spectral Density (PSD) for each channel in the raw EEG data.

    Parameters:
        raw (mne.io.Raw): The raw EEG data.
        fmin (float): Minimum frequency for PSD computation.
        fmax (float): Maximum frequency for PSD computation.

    Returns:
        freqs (ndarray): Frequencies at which the PSD is computed.
        psd_array (ndarray): PSD values for each channel, shape: (n_channels, n_freqs).
    """
    psd_array = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=2048, n_overlap=1024)
    freqs = psd_array[0]
    psd_array = psd_array[1]
    
    return freqs, psd_array

def save_psd_metrics_csv(metrics_dict, subject_id, sex, group, condition, output_path, file_name):
    """ Save global network metrics to a CSV file.
    Parameters:
    - metrics_dict : dict
        Dictionary containing the metrics to save.
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
