"""
Reproduce figures from Liu et al., Acta Cryst. A 82, 4 (2026).

Simulates a 2D WCA glass under athermal quasistatic simple shear.
Computes:
  - Local centrosymmetry F_IS and non-affine displacements  (Fig. 2 analog)
  - Burgers vector magnitude/direction and quadrupolar field  (Fig. 3 analog)
  - Parameter magnitude vs centrosymmetry quartile            (Fig. 4 analog)

The 2D WCA (Weeks-Chandler-Andersen) model with gradient-descent energy
minimization is a direct analog of the Kremer-Grest polymer glass used
in the paper. Both are athermal, jammed, and produce localized plastic
events under shear.

Run:
    python reproduce/acta_cryst_2026.py

Output:
    figures/acta_cryst_2026_glass.png
    figures/acta_cryst_2026_maps.png
    figures/acta_cryst_2026_quartile.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from sim.glass2d import (
    generate_2d_glass,
    get_nearest_neighbors,
    run_quasistatic_shear,
    local_centrosymmetry_2d,
)
from analysis.burgers import (
    distortion_field_from_positions,
    delta_distortion_field,
    distortion_tensor_to_grid,
    compute_burgers_fast,
    compute_quadrupole,
    interpolate_to_grid,
    parameter_vs_centrosymmetry_quartiles,
)

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'

# Parameters

N          = 120       # particles
PHI        = 0.82      # packing fraction
GAMMA_TOT  = 0.08      # total shear strain
D_GAMMA    = 0.01      # strain increment
N_MIN      = 120       # WCA minimization steps per increment
GRID_RES   = 35        # spatial interpolation grid
SEED       = 42


def _grid_to_particles(grid_data, positions, L, grid_res):
    """Interpolate grid field back to particle positions."""
    xi  = np.linspace(0, L, grid_res)
    zi  = np.linspace(0, L, grid_res)
    interp = RegularGridInterpolator(
        (zi, xi), grid_data, method='linear',
        bounds_error=False,
        fill_value=float(np.nanmean(grid_data)),
    )
    return interp(positions[:, ::-1])



# Simulation

def run_simulation():
    print(f'Generating WCA glass (N={N}, phi={PHI})...')
    pos, sigma, L = generate_2d_glass(N=N, packing_fraction=PHI,
                                       seed=SEED, n_min_steps=300)
    print(f'  L={L:.2f}')

    nbrs_i, _ = get_nearest_neighbors(pos, L, r_cut=1.5 * sigma)
    F_IS_i    = local_centrosymmetry_2d(pos, nbrs_i, L)
    print(f'  Initial F_IS: mean={F_IS_i.mean():.3f} std={F_IS_i.std():.3f}')

    print(f'AQS shear: gamma_total={GAMMA_TOT}, d_gamma={D_GAMMA}...')
    traj = run_quasistatic_shear(
        pos, L, sigma=sigma,
        gamma_total=GAMMA_TOT, d_gamma=D_GAMMA,
        n_min_steps=N_MIN, seed=SEED,
    )
    for step in traj:
        print(f'  gamma={step["gamma"]:.3f}  u_na={step["u_na_mag"]:.4f}')

    last     = traj[-1]
    pos_f    = last['positions']
    u_na     = last['u_na']
    u_na_mag = np.linalg.norm(u_na, axis=1)
    gamma_f  = last['gamma']

    nbrs_f, _ = get_nearest_neighbors(pos_f, L, r_cut=1.5 * sigma)
    F_IS_f    = local_centrosymmetry_2d(pos_f, nbrs_f, L)
    dF_IS     = F_IS_f - F_IS_i

    print('Computing distortion tensors...')
    e_before, _ = distortion_field_from_positions(pos,   nbrs_i, L)
    e_after,  _ = distortion_field_from_positions(pos_f, nbrs_f, L)

    # Replace non-finite values with identity
    for e in [e_before, e_after]:
        bad = ~np.isfinite(e).all(axis=(1, 2))
        e[bad] = np.eye(2)

    e_delta = delta_distortion_field(e_before, e_after)
    gx, gz, e_grid = distortion_tensor_to_grid(pos, e_delta, L, grid_res=GRID_RES)
    burgers, b_mag = compute_burgers_fast(e_grid, L, grid_res=GRID_RES, smooth_sigma=1.2)
    Q_mag, Q_dir, _ = compute_quadrupole(e_grid)

    _, _, fis_grid  = interpolate_to_grid(pos,  F_IS_i, L, grid_res=GRID_RES)
    _, _, dFis_grid = interpolate_to_grid(pos,  dF_IS,  L, grid_res=GRID_RES)

    b_at_parts = _grid_to_particles(b_mag, pos, L, GRID_RES)
    q_at_parts = _grid_to_particles(Q_mag, pos, L, GRID_RES)

    print(f'  b_mag: {b_mag.min():.4f} - {b_mag.max():.4f}')
    print(f'  u_na:  {u_na_mag.min():.4f} - {u_na_mag.max():.4f}')

    return dict(
        pos=pos, pos_f=pos_f, L=L, sigma=sigma,
        u_na=u_na, u_na_mag=u_na_mag,
        F_IS_i=F_IS_i, F_IS_f=F_IS_f, dF_IS=dF_IS,
        burgers=burgers, b_mag=b_mag,
        Q_mag=Q_mag, Q_dir=Q_dir,
        fis_grid=fis_grid, dFis_grid=dFis_grid,
        b_at_parts=b_at_parts, q_at_parts=q_at_parts,
        gamma_f=gamma_f,
    )
  
# Figures

def figure_glass(sim):
    pos=sim['pos']; pos_f=sim['pos_f']; L=sim['L']
    u_na=sim['u_na']; u_na_mag=sim['u_na_mag']
    F_IS_i=sim['F_IS_i']; dF_IS=sim['dF_IS']
    gamma_f=sim['gamma_f']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white'); ax.set_aspect('equal')

    sc1 = axes[0].scatter(pos[:, 0], pos[:, 1], c=F_IS_i,
                           cmap='RdYlBu', s=50, vmin=0.8, vmax=1.0)
    plt.colorbar(sc1, ax=axes[0], label='F_IS', fraction=0.046)
    axes[0].set_title('(A) Initial glass — centrosymmetry F_IS', fontsize=10)

    sc2 = axes[1].scatter(pos_f[:, 0], pos_f[:, 1], c=u_na_mag,
                           cmap='hot_r', s=50)
    plt.colorbar(sc2, ax=axes[1], label='|u_NA|', fraction=0.046)
    step = max(1, len(pos) // 30)
    axes[1].quiver(pos[::step, 0], pos[::step, 1],
                   u_na[::step, 0], u_na[::step, 1],
                   alpha=0.6, color='#333', scale=1.5, width=0.006)
    axes[1].set_title(f'(B) After AQS shear gamma={gamma_f:.2f}\nnon-affine displacements', fontsize=10)

    sc3 = axes[2].scatter(pos_f[:, 0], pos_f[:, 1], c=dF_IS,
                           cmap='coolwarm', s=50, vmin=-0.15, vmax=0.15)
    plt.colorbar(sc3, ax=axes[2], label='Delta F_IS', fraction=0.046)
    axes[2].set_title('(C) Change in centrosymmetry during shear', fontsize=10)

    for ax in axes:
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_xlabel('x'); ax.set_ylabel('z')

    fig.suptitle(
        f'2D WCA glass under AQS simple shear (gamma={gamma_f:.2f})\n'
        'Analog to Liu et al. Acta Cryst 2026 Fig. 2',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = 'figures/acta_cryst_2026_glass.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_maps(sim):
    L=sim['L']; burgers=sim['burgers']; b_mag=sim['b_mag']
    Q_mag=sim['Q_mag']; Q_dir=sim['Q_dir']
    fis_grid=sim['fis_grid']; dFis_grid=sim['dFis_grid']
    EXT = [0, L, 0, L]

    maps_info = [
        (b_mag,  'Burgers magnitude |b|',     'Reds',    None),
        (np.arctan2(burgers[:, :, 1], burgers[:, :, 0]),
                 'Burgers direction',           'hsv',    (-np.pi, np.pi)),
        (Q_mag,  'Quadrupole magnitude |Q|',   'Purples', None),
        (Q_dir,  'Quadrupole direction',        'RdYlGn', (-np.pi/2, np.pi/2)),
        (fis_grid,  'F_IS [centrosymmetry]',   'RdYlBu', (0.8, 1.0)),
        (dFis_grid, 'Delta F_IS (shear)',       'coolwarm', (-0.1, 0.1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('white')
    for ax, (data, title, cmap, vlim) in zip(axes.ravel(), maps_info):
        ax.set_facecolor('white')
        vmin = data.min() if vlim is None else vlim[0]
        vmax = data.max() if vlim is None else vlim[1]
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax, extent=EXT)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('z')

    fig.suptitle(
        'Geometric indicators of plasticity — 2D WCA glass under simple shear\n'
        'Reproduction of Liu et al. Acta Cryst 2026 Fig. 3',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = 'figures/acta_cryst_2026_maps.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()


def figure_quartile(sim):
    F_IS_i=sim['F_IS_i']
    u_na_mag=sim['u_na_mag']
    b_at_parts=sim['b_at_parts']
    q_at_parts=sim['q_at_parts']

    param_pairs = [
        (u_na_mag,   '|u_NA| [non-affine disp.]', '#2c3e50'),
        (b_at_parts, '|b| [Burgers magnitude]',    '#c0392b'),
        (q_at_parts, '|Q| [Quadrupole magnitude]', '#8e44ad'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('white')
    for ax, (param, lbl, col) in zip(axes, param_pairs):
        ax.set_facecolor('white')
        qc, pm, ps = parameter_vs_centrosymmetry_quartiles(
            F_IS_i, param, n_quartiles=4
        )
        ax.errorbar(np.arange(1, 5), pm, yerr=ps,
                    fmt='s-', color=col, linewidth=2.5,
                    markersize=9, capsize=5, capthick=1.5)
        ax.set_xlabel('Quartile of F_IS (centrosymmetry)', fontsize=12)
        ax.set_ylabel(lbl, fontsize=10)
        ax.set_xticks([1, 2, 3, 4])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Parameter magnitude vs centrosymmetry quartile\n'
        'Liu et al. Acta Cryst 2026 Fig. 4 analog — 2D WCA glass',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = 'figures/acta_cryst_2026_quartile.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Saved {path}')
    plt.close()

# Main


if __name__ == '__main__':
    print('=== Milestone 4: Burgers Vector and Quadrupolar Strain ===\n')
    sim = run_simulation()
    print()
    figure_glass(sim)
    figure_maps(sim)
    figure_quartile(sim)
    print('\nDone. Figures in figures/')
