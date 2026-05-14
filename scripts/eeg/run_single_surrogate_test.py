"""
Run surrogate PSI test (n_surrogates=500) for a single subject/band and save results.
Usage: python scripts/eeg/run_single_surrogate_test.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(r"D:\FY2025\Fukuyama\work place\eeg-mri-aging-analysis")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from scripts.eeg.connectivity import surrogate_psi_test_for_pair, benjamini_hochberg
from mne_connectivity.viz import plot_connectivity_circle

subject_id = 'sub-010002'
condition = 'EC'
band_pair = ('alpha', 'alpha')
band_list = ['alpha']

n_surrogates = 500
cycles = 6
overlap = 0.5
m_max_extra = 3
min_win_sec = 0.5
max_win_sec = 10.0
alpha_level = 0.05

# paths
pair_result_path = WORKSPACE_ROOT / 'data' / 'results' / 'pairwise' / subject_id
pair_result_path.mkdir(parents=True, exist_ok=True)
preproc_path = WORKSPACE_ROOT / 'data' / 'preprocessed' / subject_id
raw_path = preproc_path / f"{subject_id}_{condition}_eeg.fif"

if not raw_path.exists():
    raise FileNotFoundError(f"Preprocessed raw file not found: {raw_path}")

print('Loading raw from', raw_path)
from mne import io
raw = io.read_raw_fif(str(raw_path), preload=True)
raw.pick('eeg')
data = raw.get_data()
sfreq = raw.info['sfreq']
ch_names = raw.ch_names

print(f'Running surrogate test: subject={subject_id}, condition={condition}, band={band_pair}, n_surrogates={n_surrogates}')
observed_mean, p_matrix, sig_mask, surrogates = surrogate_psi_test_for_pair(
    data, sfreq, band_pair, band_list,
    n_surrogates=n_surrogates, cycles=cycles, overlap=overlap,
    m_max_extra=m_max_extra, min_win_sec=min_win_sec, max_win_sec=max_win_sec,
    random_state=None, alpha=alpha_level, verbose=True, show_progress=False
)

# save results
base = f"surrogate_{band_pair[0]}_{band_pair[1]}_{subject_id}_{condition}"
np.save(pair_result_path / f"{base}_observed.npy", observed_mean)
if p_matrix is not None:
    np.save(pair_result_path / f"{base}_p.npy", p_matrix)
if sig_mask is not None:
    np.save(pair_result_path / f"{base}_sigmask.npy", sig_mask)
if surrogates is not None:
    try:
        np.save(pair_result_path / f"{base}_samples.npy", surrogates)
    except Exception:
        # surrogates may be large; skip if cannot save
        pass

# if sig_mask is missing but p_matrix exists, compute BH
if sig_mask is None and p_matrix is not None:
    sig_mask = benjamini_hochberg(p_matrix, alpha=alpha_level)
    np.save(pair_result_path / f"{base}_sigmask_from_p.npy", sig_mask)

# report
n_ch = observed_mean.shape[0]
if sig_mask is None:
    print('No significance mask available.')
else:
    np.fill_diagonal(sig_mask, False)
    triu = np.triu_indices(n_ch, k=1)
    n_sig = int(np.count_nonzero(sig_mask[triu]))
    print(f'Number of significant upper-triangle edges: {n_sig}')

    # save list of significant pairs
    triu_idx = np.triu_indices(n_ch, k=1)
    sig_inds = np.where(sig_mask[triu_idx])[0]
    txt = pair_result_path / f'significant_pairs_{band_pair[0]}_{band_pair[1]}_{subject_id}_{condition}.txt'
    with open(txt, 'w', encoding='utf-8') as fh:
        fh.write('i\tj\tch_i\tch_j\tobserved\n')
        for k in sig_inds:
            i = triu_idx[0][k]
            j = triu_idx[1][k]
            fh.write(f"{i}\t{j}\t{ch_names[i]}\t{ch_names[j]}\t{observed_mean[i,j]:.6g}\n")
    print('Saved significant pair list to', txt)

    # plot connectivity circle
    adj_sig = np.where(sig_mask, observed_mean, 0.0)
    if np.count_nonzero(adj_sig) > 0:
        vmax = np.max(np.abs(observed_mean)) if np.max(np.abs(observed_mean)) != 0 else 1.0
        fig = plot_connectivity_circle(adj_sig, ch_names, title=f'{subject_id} {condition} significant psi_mean', colormap='viridis', vmin=0.0, vmax=vmax, show=False)
        img = pair_result_path / f'connectivity_circle_significant_{subject_id}_{condition}.png'
        try:
            fig.savefig(str(img), dpi=300)
            plt.close(fig)
        except Exception:
            plt.savefig(str(img), dpi=300)
            plt.close()
        print('Saved connectivity circle to', img)

print('Done.')
