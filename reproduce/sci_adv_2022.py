"""
Reproduce figures from Liu et al., Science Advances 8, eabn0681 (2022).

Generates spatial maps of local stability, centrosymmetry ratio,
normal anisotropy, and shear anisotropy for:
  - A colloidal glass aged 2 days vs 20 days  (Fig. 2 analog)
  - A glass before vs after compression        (Fig. 4 analog)
  - Histograms of parameter distributions      (Fig. 2C analog)
  - Parameter means vs centrosymmetry quartile (Fig. 3A analog)

Run:
    python reproduce/sci_adv_2022.py

Output:
    figures/sci_adv_2022_aging.png
    figures/sci_adv_2022_deformation.png
    figures/sci_adv_2022_histograms.png
    figures/sci_adv_2022_quartile.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.centrosymmetry import (
    simulate_spatial_maps,
    simulate_aging,
    simulate_deformation,
)

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

GRID = 40
SEED = 42

PARAMS = ['stability', 'centrosymmetry', 'eps_n', 'eps_s']
LABELS = [
    'C(t)  [stability]',
    'Centrosymmetry ratio',
    'eps_n  [normal anisotropy]',
    'eps_s  [shear anisotropy]',
]
CMAPS  = ['RdYlGn', 'RdYlBu', 'coolwarm', 'coolwarm']


def get_shared_vranges(m1, m2):
    return [
        [min(m1[k].min(), m2[k].min()), max(m1[k].max(), m2[k].max())]
        for k in PARAMS
    ]


def plot_four_maps(axes_row, maps, vranges):
    for ax, key, lbl, cmap, vr in zip(axes_row, PARAMS, LABELS, CMAPS, vranges):
        im = ax.imshow(maps[key], cmap=cmap, vmin=vr[0], vmax=vr[1],
                       origin='lower', aspect='equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])


def figure_aging(maps_young, maps_aged):
    vr = get_shared_vranges(maps_young, maps_aged)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    plot_four_maps(axes[0], maps_young, vr)
    plot_four_maps(axes[1], maps_aged,  vr)
    axes[0, 0].set_ylabel('(A) Aged 2 days',  fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('(B) Aged 20 days', fontsize=11, fontweight='bold')
    fig.suptitle(
        'Local stability and structure maps during aging\n'
        'Reproduction of Liu et al. Science Advances 2022 Fig. 2',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = 'figures/sci_adv_2022_aging.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_deformation(maps_young, maps_deformed):
    vr = get_shared_vranges(maps_young, maps_deformed)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    plot_four_maps(axes[0], maps_young,    vr)
    plot_four_maps(axes[1], maps_deformed, vr)
    axes[0, 0].set_ylabel('(A) Before compression', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('(B) After compression',  fontsize=11, fontweight='bold')
    fig.suptitle(
        'Local stability and structure maps during deformation\n'
        'Reproduction of Liu et al. Science Advances 2022 Fig. 4',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = 'figures/sci_adv_2022_deformation.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_histograms(maps_young, maps_aged):
    colors = [
        ('#2ecc71', '#27ae60'),
        ('#3498db', '#2980b9'),
        ('#e74c3c', '#c0392b'),
        ('#9b59b6', '#8e44ad'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    fig.patch.set_facecolor('white')
    for ax, key, lbl, (c1, c2) in zip(axes, PARAMS, LABELS, colors):
        ax.set_facecolor('white')
        ax.hist(maps_young[key].ravel(), bins=25, color=c1, alpha=0.65,
                label='Aged 2 days',  density=True)
        ax.hist(maps_aged[key].ravel(),  bins=25, color=c2, alpha=0.65,
                label='Aged 20 days', density=True)
        ax.set_xlabel(lbl, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(
        'Parameter histograms during aging — Sci Adv 2022 Fig. 2C analog',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = 'figures/sci_adv_2022_histograms.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_quartile(maps_young, maps_aged):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.patch.set_facecolor('white')
    colors_pair = [('#2ecc71', '#27ae60'), ('#3498db', '#2980b9'),
                   ('#e74c3c', '#c0392b'), ('#9b59b6', '#8e44ad')]

    for ax, target_key, tgt_lbl, (c1, c2) in zip(axes, PARAMS, LABELS, colors_pair):
        ax.set_facecolor('white')
        for maps, lbl, col in [(maps_young, 'Aged 2 days', c1),
                                (maps_aged,  'Aged 20 days', c2)]:
            cs  = maps['centrosymmetry'].ravel()
            val = maps[target_key].ravel()
            means, stds = [], []
            for q in range(4):
                lo = np.percentile(cs, q * 25)
                hi = np.percentile(cs, (q + 1) * 25)
                mask = (cs >= lo) & (cs <= hi)
                means.append(val[mask].mean())
                stds.append(val[mask].std())
            ax.errorbar(np.arange(1, 5), means, yerr=stds,
                        fmt='o-', color=col, label=lbl,
                        linewidth=2, markersize=7, capsize=4)
        ax.set_xlabel('Centrosymmetry quartile', fontsize=10)
        ax.set_ylabel(tgt_lbl, fontsize=9)
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Parameter means per centrosymmetry quartile — Sci Adv 2022 Fig. 3A analog',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = 'figures/sci_adv_2022_quartile.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


if __name__ == '__main__':
    print('=== Milestone 3: Centrosymmetry and Anisotropy Maps ===\n')

    print('Simulating glass maps...')
    maps_young    = simulate_spatial_maps(grid_size=GRID, seed=SEED,
                                          fingerprints_dict={})
    maps_aged     = simulate_aging(maps_young,    grid_size=GRID, seed=SEED + 1)
    maps_deformed = simulate_deformation(maps_young, grid_size=GRID, seed=SEED + 2)

    figure_aging(maps_young, maps_aged)
    figure_deformation(maps_young, maps_deformed)
    figure_histograms(maps_young, maps_aged)
    figure_quartile(maps_young, maps_aged)

    print('\nDone. Figures in figures/')
