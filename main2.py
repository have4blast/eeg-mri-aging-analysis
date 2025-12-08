from utils import parser
import os

from scripts.eeg.preprocess_run import preprocess_all_subjects
from scripts.eeg.connectivity import run_psi


import logging

logger = logging.getLogger(__name__)

def main():
    args = parser.get_parser()
    #preprocess_all_subjects(args)
    run_psi(args)

if __name__ == '__main__':
    main()