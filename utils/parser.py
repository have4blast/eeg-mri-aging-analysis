import argparse

def get_parser():
    """
    Create and return an ArgumentParser for EEG preprocessing.

    This parser is used to provide command-line arguments to control
    the EEG preprocessing pipeline using the MNE-Python library.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="EEG preprocessing pipeline using MNE"
    )

    # Required arguments
    # parser.add_argument('--input', type=str, required=True,
    #                     help='Path to the input EEG file (.set, .fif, .vhdr, etc.)')
    # parser.add_argument('--output', type=str, required=True,
    #                     help='Path to save the preprocessed EEG data (.fif)')
    
    parser.add_argument('--raw_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\raw\\', 
                        help='Path to the raw EEG data directory')
    parser.add_argument('--preprocess_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\preprocessed\\', 
                        help='Path to save preprocessed EEG data')
    parser.add_argument('--epochs_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\epochs\\', 
                        help='Path to save epochs data')
    parser.add_argument('--ica_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\ica\\',
                        help='Path to save ICA results and plots')
    parser.add_argument('--psd_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\psd\\', 
                        help='Path to save PSD results')
    parser.add_argument('--tfr_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\tfr\\', 
                        help='Path to save TFR results')
    parser.add_argument('--psi_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\psi\\', 
                        help='Path to save PSI results')
    parser.add_argument('--connectivity_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\connectivity\\', 
                        help='Path to save connectivity results')
    parser.add_argument('--metadata_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\metadata\\Participants_MPILMBB_LEMON.csv', 
                            help='Path to the metadata CSV file')
    parser.add_argument('--pair_result_path', type=str, default = 'D:\\FY2025\\Fukuyama\\work place\\eeg-mri-aging-analysis\\data\\results\\pairwise\\', 
                        help='Path to save pairwise results')

    # Optional preprocessing parameters
    parser.add_argument('--montage', type=str, default='standard_1005',
                        help='Montage to apply to the EEG data (default: standard_1005)')
    parser.add_argument('--low_freq', type=float, default=1.0,
                        help='Low cutoff frequency for bandpass filter (Hz)')
    parser.add_argument('--high_freq', type=float, default=40.0,
                        help='High cutoff frequency for bandpass filter (Hz)')
    #parser.add_argument('--notch_freq', type=float, default=50.0,
    #                help='Notch filter frequency (Hz), e.g., 50 or 60')
    #parser.add_argument('--resample', type=float, default=None,
    #                    help='Resample the data to given sampling rate (Hz)')
    #parser.add_argument('--set_eog', action='store_true',
    #                    help='Use default EOG channel labeling (e.g. VEOG)')

    parser.add_argument('--epoch_duration', type=float, default=2.0,
                        help='Duration of epochs in seconds (default: 2.0)')

    # ICA options
    parser.add_argument('--ica', action='store_true',
                        help='Apply ICA to remove artifacts')
    parser.add_argument('--n_components', type=float, default=0.95,
                        help='Fraction of variance to retain in ICA')
    parser.add_argument('--ica_method', type=str, default='infomax',
                        choices=['fastica', 'infomax', 'picard'],
                        help='ICA algorithm to use')

    # Misc
    parser.add_argument('--plot', action='store_true',
                        help='Plot raw and ICA components during processing')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting output files')

    # Arguments used by connectivity pipeline (used via getattr in connectivity.py)
    parser.add_argument('--same_band_only', action='store_false',
                        help='If set, compute only same-band (b,b) PSI pairs')
    parser.add_argument('--band_list', type=str, default='delta,theta,alpha,beta',
                        help='Comma-separated list of bands to analyze (default: delta,theta,alpha,beta)')
    parser.add_argument('--cycles', type=int, default=6,
                        help='Number of cycles used to determine window length')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Window overlap fraction (0.0-1.0)')
    parser.add_argument('--n_surrogates', type=int, default=5,
                        help='Number of phase-randomization surrogates to generate (default: 500)')
    parser.add_argument('--m_max_extra', type=int, default=3,
                        help='Extra margin when estimating m_max for n:m search')
    parser.add_argument('--min_win_sec', type=float, default=0.5,
                        help='Minimum window length in seconds')
    parser.add_argument('--max_win_sec', type=float, default=10.0,
                        help='Maximum window length in seconds')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random seed for surrogate generation')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for FDR correction')
    parser.add_argument('--save_surrogates', action='store_true',
                        help='Save surrogate samples to disk')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output in processing scripts')
    parser.add_argument('--show_progress', action='store_true',
                        help='Show per-surrogate progress bars when running serially (n_jobs=1)')
    parser.add_argument('--progress_position', type=int, default=0,
                        help='tqdm progress bar position (useful when stacking bars)')

    # Parallelization / progress options for surrogate computation
    parser.add_argument('--n_jobs', type=int, default=4,
                        help='Number of parallel workers for surrogate/pair processing (-1 = all cores)')
    parser.add_argument('--parallel_backend', type=str, default='loky',
                        help='Parallel backend to use (e.g. loky, threading)')

    args = parser.parse_args()

    # Convert band_list string to Python list if provided as comma-separated string
    if isinstance(args.band_list, str):
        args.band_list = [b.strip() for b in args.band_list.split(',') if b.strip()]

    return args