"""
2D glass model: WCA packing with athermal quasistatic shear.

Generates a 2D amorphous packing of N discs using:
  1. Random initial placement
  2. WCA (Weeks-Chandler-Andersen) energy minimization via gradient descent
     to produce a mechanically stable jammed configuration
  3. Athermal quasistatic shear (AQS): alternating affine strain + minimization

The WCA potential is:
    U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6] + eps   for r < 2^(1/6)*sigma
    U(r) = 0                                            for r >= 2^(1/6)*sigma

This is the standard model for jammed 2D packings (e.g. Falk & Langer 1998,
Maloney & Lemaitre 2006) and produces realistic non-affine displacements under
shear, matching the Acta Cryst 2026 polymer glass simulations qualitatively.

Reference: Liu et al., Acta Cryst. A 82, 4 (2026); Baggioli et al., PRL 127, 015501 (2021)
"""

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# WCA potential
# ---------------------------------------------------------------------------

R_CUT_FACTOR = 2.0 ** (1.0 / 6.0)   # WCA cutoff: r_cut = 2^(1/6) * sigma


def wca_energy_and_forces(positions, box_size, sigma=1.0, eps=1.0):
    """
    Compute WCA energy and forces with periodic boundary conditions.

    Parameters
    ----------
    positions : (N, 2) array
    box_size  : float
    sigma, eps : WCA parameters (sigma = particle diameter)

    Returns
    -------
    energy : float
    forces : (N, 2) array
    """
    N = len(positions)
    r_cut = R_CUT_FACTOR * sigma
    energy = 0.0
    forces = np.zeros((N, 2))

    # Build neighbor list using KDTree on periodic images
    tree = _periodic_tree(positions, box_size)
    all_pos = _periodic_images(positions, box_size)

    for i in range(N):
        idx = tree.query_ball_point(positions[i], r=r_cut)
        for j_img in idx:
            j = j_img % N
            if j <= i:
                continue
            rij = positions[i] - all_pos[j_img]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12 or r2 >= r_cut**2:
                continue
            r2_inv = (sigma**2) / r2
            r6_inv = r2_inv**3
            r12_inv = r6_inv**2
            # Energy
            energy += 4.0 * eps * (r12_inv - r6_inv) + eps
            # Force: F = -dU/dr * rij/r = 24*eps*(2*r12_inv - r6_inv)/r^2 * rij
            f_mag = 24.0 * eps * (2.0 * r12_inv - r6_inv) / r2
            fij = f_mag * rij
            forces[i] += fij
            forces[j] -= fij

    return energy, forces


def minimize_energy(positions, box_size, sigma=1.0, eps=1.0,
                    n_steps=500, lr=0.01, tol=1e-4):
    """
    Gradient descent energy minimization (steepest descent).

    Parameters
    ----------
    positions : (N, 2)
    box_size  : float
    n_steps   : int  — max gradient descent steps
    lr        : float — initial learning rate
    tol       : float — force tolerance for convergence

    Returns
    -------
    positions : (N, 2) — minimized configuration
    energies  : list of float
    """
    pos = positions.copy()
    energies = []

    for step in range(n_steps):
        E, F = wca_energy_and_forces(pos, box_size, sigma, eps)
        energies.append(E)
        f_rms = np.sqrt(np.mean(F**2))
        if f_rms < tol:
            break
        # Line search: reduce lr if energy increases
        step_size = lr / (f_rms + 1e-8)
        pos_new = (pos + step_size * F) % box_size
        E_new, _ = wca_energy_and_forces(pos_new, box_size, sigma, eps)
        if E_new < E:
            pos = pos_new
            lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.5

    return pos, energies


# ---------------------------------------------------------------------------
# Glass generation
# ---------------------------------------------------------------------------

def generate_2d_glass(N=200, packing_fraction=0.82, box_size=None,
                       seed=42, minimize=True, n_min_steps=300):
    """
    Generate a mechanically stable 2D glass via random placement + WCA minimization.

    Parameters
    ----------
    N                : int
    packing_fraction : float  — target area fraction (phi ~ 0.82 gives jammed state)
    box_size         : float or None
    seed             : int
    minimize         : bool   — run WCA minimization (True for realistic glass)
    n_min_steps      : int    — minimization steps

    Returns
    -------
    positions : (N, 2) — mechanically stable particle positions
    sigma     : float  — particle diameter (= 1.0)
    box_size  : float
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0

    if box_size is None:
        # phi = N * pi * (sigma/2)^2 / L^2  =>  L = sigma * sqrt(N*pi / (4*phi))
        box_size = sigma * np.sqrt(N * np.pi / (4.0 * packing_fraction))

    # Random initial placement (allow overlaps — minimizer will fix them)
    positions = rng.uniform(0, box_size, (N, 2))

    if minimize:
        positions, _ = minimize_energy(positions, box_size, sigma=sigma,
                                        n_steps=n_min_steps, lr=0.02)

    return positions, sigma, box_size


# ---------------------------------------------------------------------------
# Neighbor list
# ---------------------------------------------------------------------------

def get_nearest_neighbors(positions, box_size, r_cut=None):
    """
    Find nearest neighbors within r_cut using periodic KDTree.

    Parameters
    ----------
    positions : (N, 2)
    box_size  : float
    r_cut     : float or None — defaults to 1.5 * sigma (estimated from nn dist)

    Returns
    -------
    neighbor_list : list of N arrays of neighbor indices (in original array)
    nn_distances  : list of N arrays of distances
    """
    N = len(positions)
    all_pos = _periodic_images(positions, box_size)
    tree = cKDTree(all_pos)

    if r_cut is None:
        # Estimate nn distance from first shell
        _, idx = cKDTree(positions).query(positions, k=2)
        nn_dist = np.linalg.norm(positions - positions[idx[:, 1]], axis=1).mean()
        r_cut = 1.5 * nn_dist

    neighbor_list = []
    nn_distances = []

    for i in range(N):
        idx_img = np.array(tree.query_ball_point(positions[i], r=r_cut))
        if len(idx_img) == 0:
            neighbor_list.append(np.array([], dtype=int))
            nn_distances.append(np.array([]))
            continue
        orig = idx_img % N
        # Remove self
        mask = orig != i
        orig = orig[mask]; idx_img = idx_img[mask]
        dists = np.linalg.norm(all_pos[idx_img] - positions[i], axis=1)
        neighbor_list.append(orig)
        nn_distances.append(dists)

    return neighbor_list, nn_distances


# ---------------------------------------------------------------------------
# Athermal quasistatic shear
# ---------------------------------------------------------------------------

def apply_affine_shear(positions, box_size, gamma):
    """
    Apply simple shear increment gamma: u_x = gamma * z, u_z = 0.

    Returns
    -------
    pos_sheared   : (N, 2)
    affine_disp   : (N, 2) — the purely affine displacement
    """
    affine_disp = np.zeros_like(positions)
    affine_disp[:, 0] = gamma * positions[:, 1]
    pos_sheared = (positions + affine_disp) % box_size
    return pos_sheared, affine_disp


def compute_nonaffine_displacements(pos_before, pos_after, affine_disp, box_size):
    """
    u_NA = u_total - u_affine, with minimum image convention.
    """
    u_total = pos_after - pos_before
    u_total -= box_size * np.round(u_total / box_size)
    return u_total - affine_disp


def run_quasistatic_shear(positions, box_size, sigma=1.0,
                           gamma_total=0.10, d_gamma=0.005,
                           n_min_steps=150, seed=42):
    """
    Athermal quasistatic shear (AQS) protocol:
    1. Apply affine shear increment d_gamma
    2. Minimize WCA energy (produces non-affine displacements)
    3. Record state

    Parameters
    ----------
    positions   : (N, 2) — initial minimized glass
    box_size    : float
    sigma       : float  — particle diameter
    gamma_total : float  — total strain
    d_gamma     : float  — strain increment
    n_min_steps : int    — minimization steps per increment

    Returns
    -------
    trajectory : list of dicts with keys:
        'gamma', 'positions', 'u_na', 'u_na_mag', 'energy'
    """
    pos = positions.copy()
    trajectory = []
    gamma = 0.0

    while gamma < gamma_total - 1e-9:
        pos_before = pos.copy()
        gamma += d_gamma

        # Step 1: affine shear
        pos_sheared, affine = apply_affine_shear(pos, box_size, d_gamma)

        # Step 2: energy minimization — this is what generates real u_NA
        pos_min, energies = minimize_energy(
            pos_sheared, box_size, sigma=sigma,
            n_steps=n_min_steps, lr=0.015, tol=1e-3
        )

        # Step 3: non-affine displacements
        u_na = compute_nonaffine_displacements(pos_before, pos_min, affine, box_size)
        u_na_mag = np.linalg.norm(u_na, axis=1)

        pos = pos_min
        trajectory.append({
            'gamma': gamma,
            'positions': pos.copy(),
            'u_na': u_na.copy(),
            'u_na_mag': float(u_na_mag.mean()),
            'energy': float(energies[-1]) if energies else 0.0,
        })

    return trajectory


# ---------------------------------------------------------------------------
# Local centrosymmetry
# ---------------------------------------------------------------------------

def local_centrosymmetry_2d(positions, neighbor_list, box_size):
    """
    2D inversion symmetry parameter F_IS for each particle.

    F_IS = 1 - |Xi|^2 / |Xi|^2_max

    where Xi = sum_beta n_hat_{i->beta} is the force-imbalance vector.
    Perfectly centrosymmetric: Xi = 0, F_IS = 1.
    Maximally asymmetric: all neighbors same side, F_IS = 0.

    Parameters
    ----------
    positions     : (N, 2)
    neighbor_list : list of N index arrays
    box_size      : float

    Returns
    -------
    F_IS : (N,) in [0, 1]
    """
    N = len(positions)
    F_IS = np.zeros(N)

    for i in range(N):
        nbrs = neighbor_list[i]
        if len(nbrs) == 0:
            F_IS[i] = 0.5
            continue
        diff = positions[nbrs] - positions[i]
        diff -= box_size * np.round(diff / box_size)
        dists = np.linalg.norm(diff, axis=1)
        valid = dists > 1e-9
        if valid.sum() == 0:
            F_IS[i] = 0.5
            continue
        n_hat = diff[valid] / dists[valid, None]
        Xi = n_hat.sum(axis=0)
        n_nbr = float(valid.sum())
        F_IS[i] = 1.0 - np.dot(Xi, Xi) / max(n_nbr**2, 1.0)

    return np.clip(F_IS, 0, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _periodic_images(positions, box_size):
    """Stack 3x3 periodic images: (9N, 2)."""
    images = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            images.append(positions + np.array([dx * box_size, dy * box_size]))
    return np.vstack(images)


def _periodic_tree(positions, box_size):
    return cKDTree(_periodic_images(positions, box_size))
