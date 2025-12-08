import os
import gc
import numpy as np
import pandas as pd
import logging
from scripts.eeg import preprocess

from tqdm import tqdm

import mne
mne.set_log_level('WARNING')

logger = logging.getLogger(__name__)

def get_subject_ids(root_path):
    """
    Get a list of subject IDs from the EEG data directory.

    Args:
        root_path (str): Path to the EEG data directory.
    
    Returns:
        list: List of subject IDs (e.g., ['sub-010002', 'sub-010003', ...]).
    """
    subject_ids = [name for name in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, name)) and name.startswith('sub-')]
    return sorted(subject_ids)  # Sort for reproducibility

def process_subject(args, subject_id):
    """
    Perform EEG preprocessing for a single subject.

    Args:
        args: argparse.Namespace, contains paths and parameters
        subject_id: str, e.g., 'sub-010002'
    """
    try:
        logger.info(f'--- Processing subject: {subject_id} ---')

        # Load raw EEG data
        raw = preprocess.load_rawdata(args, subject_id)

        # Set the VEOG channel type
        raw.set_channel_types({'VEOG': 'eog'})

        # Apply montage
        preprocess.set_montage(args, raw)

        # Set EEG average reference
        raw = preprocess.set_eeg_reference(raw)

        # Divide into EO and EC
        raw_eo, raw_ec = preprocess.divide_conditions(raw)
        

        # Filter both conditions
        raw_eo = preprocess.filter_data(args, raw_eo)
        raw_ec = preprocess.filter_data(args, raw_ec)

        # Apply ICA if specified
        #if args.ica:
        logger.info(f'Applying ICA for subject {subject_id}')
        raw_eo, ica_eo = preprocess.apply_ica(args, raw_eo, 'EO', subject_id)
        raw_ec, ica_ec = preprocess.apply_ica(args, raw_ec, 'EC', subject_id)

        # Save preprocessed data
        preprocess.save_raw(raw_eo, args.preprocess_path, subject_id, condition='EO')
        preprocess.save_raw(raw_ec, args.preprocess_path, subject_id, condition='EC')

        logger.info(f'✔ Finished subject: {subject_id}')

    except Exception as e:
        logger.error(f'Error processing subject {subject_id}: {e}')

    finally:
        # Free memory
        #del raw, raw_eo, raw_ec
        #gc.collect()
        for var_name in ['raw', 'raw_eo', 'raw_ec']:
            if var_name in locals():
                del locals()[var_name]

def preprocess_all_subjects(args):
    """ 
    Preprocess all subjects in the EEG data directory.
    Args:
        args: argparse.Namespace, contains paths and parameters
    """
    
    subject_ids = get_subject_ids(args.raw_path)    

    for subject_id in tqdm(subject_ids, desc="Preprocessing Subjects"):
        process_subject(args, subject_id)
        #print(subject_id)

    logger.info('All subjects processed successfully.')
