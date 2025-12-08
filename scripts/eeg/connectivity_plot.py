import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from mne.viz import plot_connectivity_circle
from matplotlib import cm

def plot_save_psi_matrix(psi_matrix, ch_names, id, condition, output_path):
    """
    Plot the Phase Synchronization Index (PSI) matrix and save it as an image.
    Parameters:
        psi_matrix (ndarray): Phase Synchronization Index matrix, shape: (n_channels, n_channels).
        ch_names (list): List of channel names.
        id (str): Subject ID.
        condition (str): Condition label (e.g., 'EO', 'EC').
        output_path (str): Path to save the output image.
    """

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
    img_filename = os.path.join(subject_dir, f"{id}_{condition}.png")

    plt.savefig(img_filename, dpi=300)
    plt.close()
    print(f"Saved PSI matrix plot to {img_filename}")

def plot_save_connectivity_circle(con_matrix, ch_names, id, condition, output_path):
    """
    Plot the connectivity circle for the Phase Synchronization Index (PSI) matrix and save it as an image.
    Parameters:
        con_matrix (ndarray): Connectivity matrix, shape: (n_channels, n_channels).
        ch_names (list): List of channel names.
        id (str): Subject ID.
        condition (str): Condition label (e.g., 'EO', 'EC').
        output_path (str): Path to save the output image.
    
    """
    plt.ioff()

    n_channels = len(ch_names)
    cmap = cm.get_cmap('tab10', n_channels)
    node_colors = [cmap(i) for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plot_connectivity_circle(
        con=con_matrix,
        node_names=ch_names,
        title=f"Phase Synchronisation Index (PSI) Connectivity Circle ({id}, {condition})",
        colormap='coolwarm',
        n_lines=None,
        node_colors=node_colors,
        facecolor='white',
        textcolor='black',
        linewidth=1.5,
        fig=fig,
        ax=ax,
    )

    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists
    os.makedirs(subject_dir, exist_ok=True)

    # Save the figure
    img_filename = os.path.join(subject_dir, f"{id}_{condition}.png")
    fig.savefig(img_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PSI Connectivity Circle plot to {img_filename}")

def save_psi_matrix(psi_matrix, id, condition, output_path):
    """
    Save the n:m Phase Synchronization Index (PSI) matrix to a file.
    
    Parameters:
        psi_matrix (ndarray): n:m Phase Synchronization Index matrix, shape: (n_channels, n_channels).
        id (str): Subject ID.
        condition (str): Condition label (e.g., 'EO', 'EC').
        output_path (str): Path to save the output file.
    """
    
    subject_dir = os.path.join(output_path, id)

    # Ensure the output directory exists

    # Create the directory if it doesn't exist
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    
    # Save the matrix as a numpy file
    np.save(os.path.join(subject_dir, f"{id}_{condition}_psi_matrix.npy"), psi_matrix)