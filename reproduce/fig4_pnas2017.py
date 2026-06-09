"""
Reproduce figures from Liu et al., PNAS 114, 10344 (2017).

Simulates average symmetry spectra for five glass types and decomposes
each into FCC/BCC/ICO/RAN contributions using nonnegative least squares,
reproducing Fig. 4 and Fig. 5 of the paper.

Run:
    python reproduce/fig4_pnas2017.py

Output:
    figures/fig4_pnas2017_spectra.png
    figures/fig4_pnas2017_decomp.png
    figures/fig4_pnas2017_normalized.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim.clusters import get_all_clusters
from analysis.diffraction import (
    orientational_average, first_peak_q_range, symmetry_fingerprint,
)
from analysis.decompose import (
    build_basis, nnls_decompose_with_errors,
    simulate_mixed_ensemble, normalize_weights,
)

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

# Parameters

Q_CENTER     = 2 * np.pi * 1.23
Q_WIDTH      = Q_CENTER * 0.4
Q_VALUES     = np.linspace(Q_CENTER - Q_WIDTH, Q_CENTER + Q_WIDTH, 8)
Q_MASK       = np.ones(len(Q_VALUES), dtype=bool)
N_OR         = 20000
SEED         = 42
CACHE_PATH   = '/tmp/fingerprints_m1.pkl'

# Glass mixtures corresponding to PNAS 2017 Fig. 4 (A-E)
GLASS_MIXTURES = {
    'No additives\n(long-range repulsive)':   {'FCC': 0.35, 'BCC': 0.40, 'ICO': 0.05, 'RAN': 0.20},
    'Added salt\n(screened repulsion)':        {'FCC': 0.45, 'BCC': 0.20, 'ICO': 0.15, 'RAN': 0.20},
    'Added surfactant\n(short-range attract.)': {'FCC': 0.05, 'BCC': 0.05, 'ICO': 0.60, 'RAN': 0.30},
    'Sedimented\n(amorphous)':                 {'FCC': 0.08, 'BCC': 0.02, 'ICO': 0.00, 'RAN': 0.90},
    'Random\n(simulation)':                    {'FCC': 0.00, 'BCC': 0.00, 'ICO': 0.00, 'RAN': 1.00},
}

COLORS = {'FCC': '#1f4e9e', 'BCC': '#27ae60', 'ICO': '#c0392b', 'RAN': '#888888'}
GLASS_COLORS = ['#2c3e50', '#2980b9', '#e67e22', '#7f8c8d', '#bdc3c7']
GLASS_SHORT  = ['No add.', 'Salt', 'Surfact.', 'Sedim.', 'Random']


def load_or_compute_fingerprints():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            fps = pickle.load(f)
        print('Loaded cached fingerprints')
        return fps

    clusters = get_all_clusters()
    fps = {}
    for name in ['FCC', 'BCC', 'HCP', 'ICO', 'SC']:
        print(f'  Computing {name} ({N_OR:,} orientations)...')
        cn = orientational_average(clusters[name], Q_VALUES, n_max=12,
                                    n_orientations=N_OR, seed=SEED, batch_size=1000)
        fps[name] = symmetry_fingerprint(cn, Q_MASK)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(fps, f)
    return fps


def figure_spectra(fps, results_all):
    syms = np.arange(2, 13)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    fig.patch.set_facecolor('white')
    panel_labels = ['(A)', '(B)', '(C)', '(D)', '(E)']

    basis_fps = {k: fps[k] for k in ['FCC', 'BCC', 'ICO']}
    basis, labels = build_basis(basis_fps, include_ran=True)

    for ax, plab, (title, mixture) in zip(axes, panel_labels, GLASS_MIXTURES.items()):
        ax.set_facecolor('white')
        spectrum, _ = simulate_mixed_ensemble(fps, mixture,
                                               n_orientations=4000, noise_level=0.005)
        errors, fit, chi2r = nnls_decompose_with_errors(spectrum, basis, labels)

        ax.bar(syms, spectrum, color='#cccccc', width=0.6, alpha=0.9,
               label='Measured', edgecolor='white', zorder=2)
        ax.step(np.append(syms, syms[-1]+1) - 0.5, np.append(fit, fit[-1]),
                color='#333', linewidth=1.5, label='NNLS fit', where='post', zorder=3)

        ax.set_title(f'{plab}\n{title}', fontsize=8.5, pad=4)
        ax.set_xlabel('Symmetry n', fontsize=9)
        ax.set_xticks(syms[::2])
        ax.set_xlim(1.3, 13.2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel('Symmetry magnitude', fontsize=9)
    fig.suptitle(
        'Average symmetry spectra and NNLS decompositions\n'
        'Reproduction of Liu et al. PNAS 2017 Fig. 4',
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    path = 'figures/fig4_pnas2017_spectra.png'
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_decomp(results_all):
    comp_labels = ['FCC', 'BCC', 'ICO', 'RAN']
    x = np.arange(len(comp_labels))
    n_glasses = len(GLASS_MIXTURES)
    bar_width = 0.14

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    for gi, (title, errors) in enumerate(results_all.items()):
        best = np.array([errors.get(c, (0,0,0))[0] for c in comp_labels])
        lo   = np.array([errors.get(c, (0,0,0))[1] for c in comp_labels])
        hi   = np.array([errors.get(c, (0,0,0))[2] for c in comp_labels])
        offset = (gi - n_glasses / 2 + 0.5) * bar_width
        ax.bar(x + offset, best, bar_width * 0.9,
               color=GLASS_COLORS[gi], alpha=0.85,
               label=GLASS_SHORT[gi], edgecolor='white', linewidth=0.4)
        ax.errorbar(x + offset, best, yerr=[best-lo, hi-best],
                    fmt='none', color='#333', capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=13, fontweight='bold')
    for tick, col in zip(ax.get_xticklabels(), [COLORS[c] for c in comp_labels]):
        tick.set_color(col)
    ax.set_ylabel('Weight (arb.)', fontsize=12)
    ax.set_xlabel('Polyhedral component', fontsize=12)
    ax.set_title(
        'NNLS decomposition of colloidal glass symmetry spectra\n'
        'Reproduction of Liu et al. PNAS 2017 Fig. 5',
        fontsize=11,
    )
    ax.legend(fontsize=9, ncol=5, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = 'figures/fig4_pnas2017_decomp.png'
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_normalized(results_all):
    comp_labels = ['FCC', 'BCC', 'ICO', 'RAN']
    n_glasses = len(GLASS_MIXTURES)
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    bottoms = np.zeros(n_glasses)
    for comp in comp_labels:
        vals = []
        for title, errors in results_all.items():
            total = sum(v[0] for v in errors.values())
            vals.append(errors.get(comp, (0,0,0))[0] / max(total, 1e-9))
        ax.bar(np.arange(n_glasses), vals, bar_w,
               bottom=bottoms, color=COLORS[comp], alpha=0.85,
               label=comp, edgecolor='white', linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(np.arange(n_glasses))
    ax.set_xticklabels(GLASS_SHORT, fontsize=11)
    ax.set_ylabel('Normalized proportion', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        'Normalized polyhedral populations\n'
        'Reproduction of Liu et al. PNAS 2017 Fig. 5',
        fontsize=11,
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = 'figures/fig4_pnas2017_normalized.png'
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


if __name__ == '__main__':
    print('=== Milestone 2: NNLS Decomposition (PNAS 2017) ===\n')

    fps = load_or_compute_fingerprints()
    basis_fps = {k: fps[k] for k in ['FCC', 'BCC', 'ICO']}
    basis, labels = build_basis(basis_fps, include_ran=True)

    results_all = {}
    for title, mixture in GLASS_MIXTURES.items():
        spectrum, _ = simulate_mixed_ensemble(fps, mixture,
                                               n_orientations=4000, noise_level=0.005)
        errors, fit, chi2r = nnls_decompose_with_errors(spectrum, basis, labels)
        results_all[title] = errors
        print(f'  {title.split(chr(10))[0]}: chi2_r={chi2r:.1f}')

    figure_spectra(fps, results_all)
    figure_decomp(results_all)
    figure_normalized(results_all)

    print('\nDone. Figures in figures/')
