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

    args = parser.parse_args()

    return args