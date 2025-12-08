import os
import os.path as op
import matplotlib.pyplot as plt

import mne

from tqdm import tqdm

mne.set_log_level('WARNING')

def load_rawdata(args, id):
    """
    Load EEG data from a specified directory.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        id (str): Identifier for the subject, used to construct the file path.
    Returns:
        raw (mne.io.Raw): The loaded EEG data as a Raw object.
    Raises:
        FileNotFoundError: If the specified EEG file does not exist.
        Exception: If there is an error loading the EEG data.
    """
    raw_dir = args.raw_path + id + '\\RSEEG\\'  # Path to the raw EEG Data folder
    vhdr_file = op.join(raw_dir, id + '.vhdr')  # Path to the raw EEG header file

    # File existence check
    if not op.exists(vhdr_file):
        print(f"File not found: {vhdr_file}. Skipping subject {id}.")
        return None

    try:
        raw = mne.io.read_raw_brainvision(vhdr_file, misc='auto')  # Load BrainVision data
        raw.load_data()  # Load the data into memory
        return raw
    except Exception as e:
        print(f"Error loading {vhdr_file}: {e}. Skipping subject {id}.")
        return None

def set_montage(args, raw):
    """ Set the standard montage for EEG channels in the raw data.
    Args:
        raw (mne.io.Raw): The loaded EEG data.
    Returns:
        None: The montage is applied directly to the raw data.
    """ 
    # Create standard montage
    montage = mne.channels.make_standard_montage(args.montage)

    # Apply montage to EEG channels only
    # Since VEOG is not part of the standard montage, allow missing channel info
    raw.set_montage(montage, on_missing='raise')

def set_eeg_reference(raw):
    """ Set the EEG reference to average across all channels. 
    Args:
        raw (mne.io.Raw): The loaded EEG data.
    Returns:
        raw (mne.io.Raw): The EEG data with average reference applied.
    """
    raw.set_eeg_reference('average', projection=True).apply_proj()

    return raw

def divide_conditions(raw):
    """    
    Divide the EEG data into blocks based on conditions (Eyes Open and Eyes Closed). 

    Args:
        raw (mne.io.Raw): The loaded EEG data.
    Returns:
        raw_eo (mne.io.Raw): Raw data for Eyes Open condition.
        raw_ec (mne.io.Raw): Raw data for Eyes Closed condition. 
    """

    # Extract events and event_id dictionary from annotations in the raw data
    try:
        events, event_id = mne.events_from_annotations(raw)
    except Exception as e:
        print(f"Could not extract events: {e}")
        return None, None

    fs = int(raw.info['sfreq'])

    eo_blocks = []
    ec_blocks = []

    # Get numeric event codes
    eo_code = event_id.get('Stimulus/S200', -1)  # Eyes Open
    ec_code = event_id.get('Stimulus/S210', -1)  # Eyes Closed

    for i, event in tqdm(enumerate(events[:-1]), total=len(events) - 1, desc="Splitting Conditions"):
        this_sample = event[0]
        this_code = event[2]
        
        next_sample = events[i + 1][0]
        next_code = events[i + 1][2]

        t_start = this_sample / fs
        t_end = next_sample / fs

        # Only include block if the same condition continues (no switch)
        if this_code == eo_code and next_code == eo_code:
            eo_blocks.append(raw.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False))
        elif this_code == ec_code and next_code == ec_code:
            ec_blocks.append(raw.copy().crop(tmin=t_start, tmax=t_end, include_tmax=False))
        # Transitions (e.g., EO → EC or EC → EO) are skipped

    # Concatenate all blocks
    raw_eo = mne.concatenate_raws(eo_blocks) if eo_blocks else None
    raw_ec = mne.concatenate_raws(ec_blocks) if ec_blocks else None

    return raw_eo, raw_ec

# def divide_conditions_auto_margin(raw, min_block_sec=0.2):
#     """    
#     Divide EEG data into EO/EC blocks while automatically handling edges.
    
#     Args:
#         raw (mne.io.Raw): Loaded EEG data.
#         min_block_sec (float): Minimum duration of a block to include (seconds).
        
#     Returns:
#         raw_eo, raw_ec (mne.io.Raw or None): Concatenated raw blocks per condition.
#     """
#     try:
#         events, event_id = mne.events_from_annotations(raw)
#     except Exception as e:
#         print(f"Could not extract events: {e}")
#         return None, None

#     fs = raw.info['sfreq']
#     data_end = raw.times[-1]

#     eo_blocks, ec_blocks = [], []

#     eo_code = event_id.get('Stimulus/S200', -1)
#     ec_code = event_id.get('Stimulus/S210', -1)

#     for i, event in enumerate(events[:-1]):
#         t_start = event[0] / fs
#         t_end = events[i + 1][0] / fs
#         this_code = event[2]
#         next_code = events[i + 1][2]

#         # Automatically shrink margins if near edges
#         t_start_safe = max(t_start, 0)
#         t_end_safe = min(t_end, data_end)

#         block_duration = t_end_safe - t_start_safe
#         if block_duration < min_block_sec:
#             continue  # Skip very short blocks

#         if this_code == eo_code and next_code == eo_code:
#             eo_blocks.append(raw.copy().crop(tmin=t_start_safe, tmax=t_end_safe, include_tmax=False))
#         elif this_code == ec_code and next_code == ec_code:
#             ec_blocks.append(raw.copy().crop(tmin=t_start_safe, tmax=t_end_safe, include_tmax=False))

#     raw_eo = mne.concatenate_raws(eo_blocks) if eo_blocks else None
#     raw_ec = mne.concatenate_raws(ec_blocks) if ec_blocks else None

#     return raw_eo, raw_ec


def filter_data(args, raw):
    """ Apply band-pass filtering to the EEG data.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        raw (mne.io.Raw): The loaded EEG data.
    Returns:
        filtered_raw (mne.io.Raw): The filtered EEG data.
    """

    # Define IIR filter parameters: 8th-order Butterworth filter
    iir_params = dict(order=8, ftype='butter')

    # Apply a band-pass filter from 1 to 45 Hz using zero-phase filtering
    filtered_raw = raw.copy().filter(
        l_freq=args.low_freq,              # Lower cutoff frequency (Hz)
        h_freq=args.high_freq,             # Upper cutoff frequency (Hz)
        method='iir',           # Use IIR filtering
        iir_params=iir_params,  # Specify filter order and type
        phase='zero',           # Apply zero-phase (forward-backward) filtering to avoid phase distortion
        verbose=True            # Print filtering information
    )
    return filtered_raw

def apply_ica(args, raw, condition, id, random_state=97):

    # ICA fitting for EO (Eyes Open) or EC (Eyes Closed)
    ica = mne.preprocessing.ICA(n_components=args.n_components, method=args.ica_method, random_state=random_state)
    ica.fit(raw)

    # Detect and mark EOG artifacts (blink/movement)
    eog_inds, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_inds  # mark bad components

    # Plot the components and scores for inspection
    path = args.ica_path + id + '\\' + condition + '\\'
    save_ica_plots(ica, eog_scores, condition, id, path)

    # Apply ICA to remove marked components
    raw_clean = ica.apply(raw.copy())

    return raw_clean, ica

def save_ica_plots(ica, eog_scores, condition_name, id, output_dir):
    """
    Save ICA components and EOG correlation scores as images.

    Parameters:
    - ica : mne.preprocessing.ICA object
        The fitted ICA object.
    - eog_scores : array-like
        Scores computed by ica.find_bads_eog().
    - condition_name : str
        Condition label used in file names (e.g., 'EO', 'EC').
    - output_dir : str
        Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot ICA components
    fig_components = ica.plot_components(title=f'ICA Components - {id} - {condition_name}', show=False)
    fig_components.savefig(os.path.join(output_dir, f'ica_components_{id}_{condition_name}.png'), dpi=300)

    # Plot EOG correlation scores
    fig_scores = ica.plot_scores(eog_scores, title=f'EOG correlation scores - {id} - {condition_name}', show=False)
    fig_scores.savefig(os.path.join(output_dir, f'ica_scores_{id}_{condition_name}.png'), dpi=300)

    # Close figures to free memory
    plt.close(fig_components)
    plt.close(fig_scores)

def save_raw(raw, output_path, id, condition):
    """
    Save the preprocessed EEG data to a specified path.
    Args:
        raw (mne.io.Raw): The preprocessed EEG data.
        output_path (str): Directory to save the preprocessed data.
        id (str): Identifier for the subject.
        condition (str): Condition label (e.g., 'EO', 'EC').
    Returns:
        None: The function saves the raw data in FIF format.
    """
    
    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists
    #os.makedirs(os.path.dirname(subject_dir), exist_ok=True)
    os.makedirs(subject_dir, exist_ok=True)

    # Construct the full path for saving the raw data
    path = os.path.join(subject_dir, f"{id}_{condition}_eeg.fif")

    # Save the raw data in FIF format
    raw.save(path, overwrite=True)

    # Print confirmation message
    print(f"Preprocessed data saved to {path}")


def compute_csd(raw):
    """ Compute the current source density (CSD) from the raw EEG data.
    Args: 
        raw (mne.io.Raw): The loaded EEG data.
    Returns:
        raw_csd (mne.io.Raw): The EEG data with CSD applied.
    """
    raw_csd = mne.preprocessing.compute_current_source_density(raw)

    return raw_csd
