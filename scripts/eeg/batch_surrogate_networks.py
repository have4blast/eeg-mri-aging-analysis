"""
Batch surrogate network analysis for multiple subjects and frequency bands.

This script:
- Iterates over a list/range of subjects (default: sub-010002 .. sub-010036)
- For each subject and band (default: alpha), loads available pairwise pickle or observed arrays
  and runs surrogate_psi_test_for_pair with n_surrogates (default 500) if necessary
- Computes significance mask (from surrogate outputs or saved p-matrix) and builds an
  adjacency keeping only significant edges
- Computes network metrics with analyze_network_from_psi and saves per-subject metrics
- Saves adjacency matrices, significant-pair lists, and per-subject CSVs
- Aggregates global metrics across subjects and performs a simple group comparison
  (two-group t-test) using the 'group' field from metadata when available

Usage (from workspace root):
    python scripts/eeg/batch_surrogate_networks.py --n_surrogates 500

"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ensure workspace root is importable
WORKSPACE_ROOT = Path(r"D:\FY2025\Fukuyama\work place\eeg-mri-aging-analysis")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from scripts.eeg.connectivity import (
    surrogate_psi_test_for_pair,
    analyze_network_from_psi,
    benjamini_hochberg,
    get_subject_metadata,
)

from types import SimpleNamespace


def subject_list_from_range(start_id, end_id):
    """Generate list like sub-010002 .. sub-010036 inclusive given zero-padded numbers."""
    def parse(s):
        if s.startswith('sub-'):
            return int(s.split('-')[1])
        return int(s)
    s = parse(start_id)
    e = parse(end_id)
    lst = [f"sub-{i:06d}" for i in range(s, e+1)]
    return lst


def load_pair_pickle(pair_path):
    import pickle
    try:
        with open(pair_path, 'rb') as pf:
            return pickle.load(pf)
    except Exception:
        return None


def run_for_band(subjects, band_pair, args):
    band_a, band_b = band_pair
    band_label = f"{band_a}_{band_b}"
    out_rows = []

    for subject_id in tqdm(subjects, desc=f"Processing {band_a}-{band_b}"):
        subj_out_dir = Path(args.pair_result_path) / subject_id
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        # try to load pairwise pickle
        pkl_name = f"psi_{band_a}_{band_b}_{subject_id}_{args.condition}.pkl"
        pkl_path = subj_out_dir / pkl_name
        pair_res = None
        if pkl_path.exists():
            pair_res = load_pair_pickle(pkl_path)

        # try other pickle pattern used by older pipeline
        if pair_res is None:
            alt_pkl = subj_out_dir / f"{band_a}_{band_b}_{subject_id}_{args.condition}.pkl"
            if alt_pkl.exists():
                pair_res = load_pair_pickle(alt_pkl)

        # try to get psi_mean or observed_mean
        psi_mean = None
        if isinstance(pair_res, dict):
            psi_mean = pair_res.get('psi_mean') or pair_res.get('surrogate_observed') or pair_res.get('observed_mean')

        # if not available, try loading saved observed npy
        obs_npy = subj_out_dir / f"surrogate_{band_a}_{band_b}_{subject_id}_{args.condition}_observed.npy"
        if psi_mean is None and obs_npy.exists():
            psi_mean = np.load(str(obs_npy))

        # if still None, attempt to run surrogate test (needs raw data passed in proper format)
        # prefer to use surrogate_psi_test_for_pair if available in pair_res or saved arrays
        pmat = None
        sig_mask = None

        if isinstance(pair_res, dict):
            pmat = pair_res.get('surrogate_p') or pair_res.get('p_matrix')
            sig_mask = pair_res.get('surrogate_sigmask') or pair_res.get('sig_mask')

        # if pmat or sig_mask not available but observed exists, run surrogate test
        if (psi_mean is None) or (sig_mask is None and pmat is None):
            # try to find data and sfreq from preprocessed files
            try:
                from mne import io
                preprocess_path = WORKSPACE_ROOT / 'data' / 'preprocessed'
                raw_path = preprocess_path / subject_id / f"{subject_id}_{args.condition}_eeg.fif"
                if raw_path.exists():
                    raw = io.read_raw_fif(str(raw_path), preload=True)
                    raw.pick('eeg')
                    data = raw.get_data()
                    sfreq = raw.info['sfreq']
                    # run surrogate test
                    print(f"Running surrogate test for {subject_id} ({band_a}-{band_b}) n={args.n_surrogates} ...")
                    observed_mean, p_matrix, sig_mask, surrogates = surrogate_psi_test_for_pair(
                        data, sfreq, (band_a, band_b), args.band_list,
                        n_surrogates=args.n_surrogates, cycles=args.cycles, overlap=args.overlap,
                        m_max_extra=args.m_max_extra, min_win_sec=args.min_win_sec,
                        max_win_sec=args.max_win_sec, random_state=args.random_state,
                        alpha=args.alpha, verbose=args.verbose,
                        show_progress=False
                    )
                    psi_mean = observed_mean
                    pmat = p_matrix
                    # save quick results
                    np.save(str(subj_out_dir / f"surrogate_{band_a}_{band_b}_{subject_id}_{args.condition}_observed.npy"), observed_mean)
                    np.save(str(subj_out_dir / f"surrogate_{band_a}_{band_b}_{subject_id}_{args.condition}_p.npy"), p_matrix)
                    np.save(str(subj_out_dir / f"surrogate_{band_a}_{band_b}_{subject_id}_{args.condition}_sigmask.npy"), sig_mask)
                else:
                    print(f"Raw file not found for {subject_id}; skipping surrogate run and trying existing psi_mean/pfiles.")
            except Exception as e:
                print(f"Surrogate run failed for {subject_id}: {e}")

        # if pmat exists and no sig_mask, apply BH
        if sig_mask is None and pmat is not None:
            try:
                sig_mask = benjamini_hochberg(pmat, alpha=args.alpha)
            except Exception:
                sig_mask = np.array(pmat) < args.alpha

        if psi_mean is None:
            print(f"psi_mean not available for {subject_id}; skipping")
            continue

        psi_mean = np.array(psi_mean)
        # ensure square
        if psi_mean.ndim != 2 or psi_mean.shape[0] != psi_mean.shape[1]:
            print(f"psi_mean for {subject_id} has wrong shape {psi_mean.shape}; skipping")
            continue
        n_ch = psi_mean.shape[0]

        # if no sig_mask, apply proportional threshold
        if sig_mask is None:
            prop = args.top_prop
            triu_idx = np.triu_indices(n_ch, k=1)
            vals = np.abs(psi_mean[triu_idx])
            cutoff = np.quantile(vals, 1 - prop) if vals.size>0 else 0.0
            sig_mask = np.abs(psi_mean) >= cutoff

        sig_mask = np.array(sig_mask, dtype=bool)
        np.fill_diagonal(sig_mask, False)

        adj_sig = np.where(sig_mask, psi_mean, 0.0)

        # compute metrics
        try:
            metrics = analyze_network_from_psi(adj_sig, mode='weighted')
        except Exception as e:
            print(f"analyze_network_from_psi failed for {subject_id}: {e}")
            metrics = {
                'density': float(np.count_nonzero(adj_sig) / adj_sig.size),
                'node_strength': dict(enumerate(np.sum(np.abs(adj_sig), axis=1)))
            }

        # prepare row for global metrics summary
        meta = get_subject_metadata(subject_id, str(WORKSPACE_ROOT / 'metadata' / 'Participants_MPILMBB_LEMON.csv'))
        row = {
            'subject_id': subject_id,
            'band': f"{band_a}-{band_b}",
            'density': metrics.get('density'),
            'global_efficiency': metrics.get('global_efficiency'),
            'local_efficiency': metrics.get('local_efficiency'),
            'n_nodes': int(metrics.get('n_nodes', n_ch)),
            'sex': meta.get('sex', 'unknown'),
            'group': meta.get('group', 'unknown')
        }
        out_rows.append(row)

        # save per-subject files
        # Save per-subject CSV into a group-specific folder under output_root/band_label/<group>/
        group_label = meta.get('group', 'unknown')
        group_dir = Path(args.output_root) / band_label / str(group_label)
        group_dir.mkdir(parents=True, exist_ok=True)
        subj_csv = group_dir / f"network_metrics_{band_a}_{band_b}_{subject_id}_{args.condition}.csv"
        pd.DataFrame([row]).to_csv(str(subj_csv), index=False)

        # save adjacency and sigmask
        np.save(subj_out_dir / f"psi_mean_{subject_id}_{args.condition}_adj_sig_{band_a}_{band_b}.npy", adj_sig)
        np.save(subj_out_dir / f"psi_mean_{subject_id}_{args.condition}_sigmask_{band_a}_{band_b}.npy", sig_mask)

        # save significant pair list
        triu = np.triu_indices(n_ch, k=1)
        sig_idx = np.where(sig_mask[triu])[0]
        pairs = []
        for k in sig_idx:
            i = triu[0][k]
            j = triu[1][k]
            pairs.append((i, j, float(psi_mean[i,j])))
        txtp = subj_out_dir / f'significant_pairs_{band_a}_{band_b}_{subject_id}_{args.condition}.txt'
        with open(txtp, 'w', encoding='utf-8') as fh:
            fh.write('i\tj\tvalue\n')
            for i,j,v in pairs:
                fh.write(f"{i}\t{j}\t{v:.6g}\n")

    # aggregate results to dataframe
    if len(out_rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    return df


def group_compare_and_save(df, args, band_label):
    # group by 'group' column
    out_dir = Path(args.output_root) / band_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # save aggregated metrics (overall)
    agg_csv = out_dir / f'global_metrics_all_subjects_{band_label}.csv'
    df.to_csv(str(agg_csv), index=False)

    # also save per-group aggregated CSVs into group subfolders
    for g, gdf in df.groupby('group'):
        safe_g = str(g) if g is not None else 'unknown'
        gd = out_dir / safe_g
        gd.mkdir(parents=True, exist_ok=True)
        gcsv = gd / f'global_metrics_{band_label}_{safe_g}.csv'
        gdf.to_csv(str(gcsv), index=False)

    # simple two-group comparison if two unique groups present
    groups = df['group'].fillna('unknown').unique()
    stats = []
    if len(groups) == 2:
        from scipy import stats as sps
        g0, g1 = groups[0], groups[1]
        df0 = df[df['group'] == g0]
        df1 = df[df['group'] == g1]
        metrics_to_test = ['density', 'global_efficiency', 'local_efficiency']
        for m in metrics_to_test:
            a = df0[m].dropna()
            b = df1[m].dropna()
            if len(a) < 2 or len(b) < 2:
                t, p = np.nan, np.nan
            else:
                t, p = sps.ttest_ind(a, b, equal_var=False, nan_policy='omit')
            stats.append({'metric': m, 'group0': g0, 'group1': g1, 't': float(t) if not np.isnan(t) else None, 'p': float(p) if not np.isnan(p) else None, 'n0': len(a), 'n1': len(b)})
    else:
        # fallback: compute group means and counts
        stats_df = df.groupby('group')[['density','global_efficiency','local_efficiency']].agg(['mean','std','count'])
        stats_df.to_csv(str(out_dir / 'group_summary_stats.csv'))
        print('Saved group summary stats to', out_dir / 'group_summary_stats.csv')
        return

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(str(out_dir / f'group_comparison_ttests_{band_label}.csv'), index=False)
    # also save stats to band/group subfolders for convenience
    for s in stats:
        g0 = s.get('group0')
        g1 = s.get('group1')
    stats_df.to_csv(str(out_dir / f'group_comparison_ttests_{band_label}.csv'), index=False)
    print('Saved group comparison results to', out_dir / f'group_comparison_ttests_{band_label}.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='sub-010002')
    parser.add_argument('--end', default='sub-010036')
    parser.add_argument('--condition', default='EO')
    parser.add_argument('--pair_result_path', default=str(WORKSPACE_ROOT / 'data' / 'results' / 'pairwise'))
    parser.add_argument('--output_root', default=str(WORKSPACE_ROOT / 'results' / 'networks'))
    parser.add_argument('--n_surrogates', type=int, default=500)
    parser.add_argument('--band_list', nargs='+', default=['alpha'])
    parser.add_argument('--cycles', type=float, default=6)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--m_max_extra', type=int, default=3)
    parser.add_argument('--min_win_sec', type=float, default=0.5)
    parser.add_argument('--max_win_sec', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--top_prop', type=float, default=0.10, help='proportional threshold when no p-matrix available')

    args = parser.parse_args()
    # convert to namespace to pass to functions
    args = SimpleNamespace(**vars(args))

    subjects = subject_list_from_range(args.start, args.end)

    # process only alpha-alpha for now; build band pairs
    band_pairs = []
    for band in args.band_list:
        band_pairs.append((band, band))

    for bp in band_pairs:
        df = run_for_band(subjects, bp, args)
        if df.empty:
            print(f'No results for band pair {bp}.')
            continue
        band_label = f"{bp[0]}_{bp[1]}"
        group_compare_and_save(df, args, band_label)

    print('Done.')


if __name__ == '__main__':
    main()
