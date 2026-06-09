"""
Reproduce Fig. 3 of Liu et al., PRL 116, 205501 (2016).

Projected bond-orientational order parameter fingerprints for
FCC, BCC, HCP, ICO, SC clusters averaged over random orientations.

Run:
    python reproduce/fig3_prl2016.py
Output:
    figures/fig3_prl2016.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sim.clusters import get_all_clusters
from analysis.diffraction import (
    orientational_average,
    first_peak_q_range,
    symmetry_fingerprint,
)

# Parameters — match PRL 2016 as closely as possible

# q range: first diffraction peak for nn distance = 1
# Ehrenfest: q_peak ~ 2*pi / d_nn * 1.23
Q_CENTER = 2 * np.pi * 1.23   # ~ 7.73 for d_nn = 1
Q_WIDTH  = Q_CENTER * 0.4     # integrate over ±20% of peak

N_Q      = 80
Q_MIN    = Q_CENTER - Q_WIDTH
Q_MAX    = Q_CENTER + Q_WIDTH
q_values = np.linspace(Q_MIN, Q_MAX, N_Q)

N_ORIENTATIONS = 40000   # good convergence; use 160000 for publication
N_MAX = 12
SEED  = 42

# Cluster display order and colors matching PRL 2016 style
CLUSTER_ORDER  = ['FCC', 'HCP', 'ICO', 'BCC', 'SC']
CLUSTER_COLORS = {
    'FCC': '#1f4e9e',
    'HCP': '#e07b00',
    'ICO': '#c0392b',
    'BCC': '#27ae60',
    'SC':  '#8e44ad',
}
CLUSTER_LABELS = {
    'FCC': 'fcc',
    'HCP': 'hcp',
    'ICO': 'icos',
    'BCC': 'bcc',
    'SC':  'sc',
}

# Compute fingerprints

def compute_fingerprints(n_orientations=N_ORIENTATIONS):
    clusters = get_all_clusters()
    q_mask = first_peak_q_range(q_values, Q_CENTER, Q_WIDTH)
    fingerprints = {}

    for name in CLUSTER_ORDER:
        print(f"Computing {name} ({n_orientations} orientations)...")
        coords = clusters[name]
        cn_avg = orientational_average(
            coords, q_values,
            n_max=N_MAX,
            n_orientations=n_orientations,
            seed=SEED,
            verbose=True,
        )
        fp = symmetry_fingerprint(cn_avg, q_mask)
        fingerprints[name] = fp
        print(f"  {name} done. Peak symmetry: n={np.argmax(fp[1:])+1}, val={fp[1:].max():.4f}")

    return fingerprints

# Plotting — Fig 3 style: bar charts, one panel per cluster

def plot_fingerprints(fingerprints, save_path='figures/fig3_prl2016.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_clusters = len(CLUSTER_ORDER)
    symmetries = np.arange(2, N_MAX + 1)   # n = 2..12, skip n=0,1

    fig, axes = plt.subplots(
        1, n_clusters,
        figsize=(13, 3.2),
        sharey=False,
    )
    fig.patch.set_facecolor('white')

    for ax, name in zip(axes, CLUSTER_ORDER):
        fp = fingerprints[name]
        vals = fp[2:]   # n=2..12

        color = CLUSTER_COLORS[name]
        ax.bar(symmetries, vals, color=color, width=0.7, alpha=0.85,
               edgecolor='white', linewidth=0.5)

        ax.set_title(CLUSTER_LABELS[name], fontsize=13, fontweight='bold',
                     color=color)
        ax.set_xlabel('Symmetry $n$', fontsize=10)
        ax.set_xticks(symmetries)
        ax.set_xlim(1.3, 13.2)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')

        # annotate dominant symmetry
        peak_n = symmetries[np.argmax(vals)]
        peak_v = vals.max()
        ax.annotate(f'$n$={peak_n}', xy=(peak_n, peak_v),
                    xytext=(peak_n + 0.6, peak_v * 0.95),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    axes[0].set_ylabel(r'$\langle c_n / c_0 \rangle$', fontsize=11)

    fig.suptitle(
        'Projected BOO fingerprints (Liu et al. PRL 2016, Fig. 3)\n'
        f'Orientational average over {N_ORIENTATIONS:,} rotations',
        fontsize=11, y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white')
    print(f"\nSaved: {save_path}")
    return fig


# Convergence check plot (Fig 2c analog)

def plot_convergence(save_path='figures/fig3_convergence_check.png'):
    """
    Show that the FCC fingerprint converges with number of orientations.
    Validates the orientational averaging routine.
    """
    from analysis.diffraction import convergence_check

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    clusters = get_all_clusters()
    q_mask = first_peak_q_range(q_values, Q_CENTER, Q_WIDTH)

    counts = [1000, 10000, 40000]
    print("Running convergence check for FCC...")
    results = convergence_check(
        clusters['FCC'], q_values, q_mask,
        n_max=N_MAX,
        orientation_counts=counts,
        seed=SEED,
    )

    symmetries = np.arange(2, N_MAX + 1)
    colors_conv = ['#aec6cf', '#5b8fa8', '#1f4e9e']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    width = 0.25
    offsets = np.linspace(-width, width, len(counts))

    for offset, (n_or, color) in zip(offsets, zip(counts, colors_conv)):
        fp = results[n_or]
        vals = fp[2:]
        ax.bar(symmetries + offset, vals, width=width * 0.9,
               color=color, alpha=0.9, label=f'{n_or:,} orientations',
               edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Symmetry $n$', fontsize=11)
    ax.set_ylabel(r'$\langle c_n / c_0 \rangle$', fontsize=11)
    ax.set_xticks(symmetries)
    ax.set_xlim(1.3, 13.2)
    ax.set_title('FCC convergence check (analog to PRL 2016 Fig. 2c)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    return fig

# Main
if __name__ == '__main__':
    print("=== Milestone 1: Projected BOO Fingerprints ===\n")

    # Convergence check first
    plot_convergence()

    # Main figure
    fingerprints = compute_fingerprints()
    plot_fingerprints(fingerprints)

    print("\nDone. Check figures/")
